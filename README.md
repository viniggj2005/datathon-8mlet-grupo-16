# Datathon 8MLET — Grupo 16

## Otimização Adaptativa de Canal de Contato Bancário

## 1. Visão do problema

Instituições financeiras digitais precisam decidir, para cada cliente elegível, qual
canal, oferta ou próximo passo apresentar em uma campanha. Regras fixas (sempre o
mesmo canal) e testes A/B longos desperdiçam tráfego e demoram para reagir a
mudanças de contexto.

Este projeto implementa uma **política adaptativa de decisão** (multi-armed bandit
contextual) que escolhe, para cada cliente, o **canal de contato com maior
probabilidade de conversão** — Celular ou Telefone Fixo — equilibrando exploração
(testar canais menos óbvios para continuar aprendendo) e explotação (usar o canal
que o modelo já sabe que converte mais).

A solução compara essa política adaptativa contra um **baseline determinístico**
(sempre usar o canal historicamente melhor) e demonstra o ganho de conversão obtido.

## 2. Base de dados

- **Fonte:** [Bank Marketing Dataset — Kaggle (henriqueyamahata)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- **Arquivo utilizado:** `bank-additional-full.csv` (41.188 registros, 21 colunas)
- **Alvo (target):** `y` — se o cliente aderiu ao depósito a prazo (`yes`/`no`), usado
  como proxy de conversão
- **Licença/uso:** dataset público e anonimizado, sem identificadores pessoais,
  renda, gênero ou raça. Usado apenas para fins acadêmicos de treinamento de modelo.
- **Base legal / minimização / retenção:** não há dados reais de clientes; o dataset
  é uma amostra pública já anonimizada disponibilizada pelo autor original no
  Kaggle. Nenhum dado é armazenado além do necessário para treino e avaliação do
  modelo neste repositório.
- **Colunas removidas por vazamento/redundância:** `duration` (vazamento temporal —
  só é conhecida após a ligação acontecer), `emp.var.rate` e `euribor3m` (altamente
  correlacionadas com `nr.employed`, mantida como representante do grupo).

## 3. Estrutura do repositório

```
datathon-8mlet-grupo-16/
├── README.md
├── requirements.txt
├── .gitignore
└── datathon.ipynb   # EDA, baseline, modelo adaptativo, golden set, serviço e MLflow
```

## 4. Instruções de execução

```bash
# 1. Clonar o repositório
git clone https://github.com/viniggj2005/datathon-8mlet-grupo-16.git
cd datathon-8mlet-grupo-16

# 2. Criar e ativar um ambiente virtual (opcional, recomendado)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Rodar o notebook
jupyter notebook datathon.ipynb
```

Principais dependências (`requirements.txt`):

```
mlflow
kagglehub
numpy
pandas
seaborn
matplotlib
scikit-learn
```

O carregamento dos dados é feito diretamente via `kagglehub`, então não é
necessário baixar o CSV manualmente.

## 5. Estratégia algorítmica

| Componente | Escolha | Justificativa |
|---|---|---|
| Baseline | Regra fixa (melhor canal histórico) | Sempre contatar via **Celular**, canal com maior taxa de conversão histórica (14,74% vs 5,23% do Telefone) |
| Modelo de contexto | `RandomForestClassifier` (n_estimators=100, max_depth=5) | Aprende a probabilidade de conversão de cada cliente em cada canal, a partir do seu perfil (idade, profissão, educação, histórico de contato, etc.) |
| Algoritmo adaptativo | **Epsilon-Greedy contextual** (ε = 0,10) | Em 90% dos casos, explora o canal com maior probabilidade prevista pelo modelo (explotação); em 10% dos casos, sorteia um canal aleatoriamente (exploração), para continuar aprendendo e evitar convergência prematura |

Optamos por Epsilon-Greedy em vez de Thompson Sampling por sua simplicidade de
implementação e interpretação direta do trade-off exploração/explotação via um
único parâmetro (ε), suficiente para o escopo deste desafio. O contexto entra na
decisão através do modelo de RandomForest, que prevê a probabilidade de conversão
por canal a partir das features do cliente — a política então usa essa
probabilidade, e não uma regra fixa, para decidir.

## 6. Resultados (Baseline vs. Algoritmo Adaptativo)

Avaliação feita via **offline evaluation (Replay Method)**: só contam os registros
em que a ação escolhida pelo algoritmo coincide com a ação realmente registrada no
histórico.

| Métrica | Valor |
|---|---|
| Taxa de conversão — Baseline (regra fixa) | 14,74% |
| Taxa de conversão — Epsilon-Greedy contextual | 16,11% |
| Ganho relativo | **+9,29%** |

O algoritmo adaptativo superou o baseline fixo, validando a abordagem.

## 7. Golden Set (casos de teste)

Um conjunto de 5 clientes sorteados aleatoriamente foi usado para validar
manualmente se as recomendações do modelo fazem sentido dado o perfil de cada um
(idade, profissão, escolaridade, quantidade de contatos na campanha e contato
anterior). Ver notebook, seção "Análise do Golden Set", para o detalhamento
caso a caso.

## 8. Serviço / Interface (Etapa 5)

A função `recomendar_canal(idade, profissao, educacao, sucesso_anterior,
forcar_exploracao)`, disponível no notebook, recebe o perfil de um cliente e
retorna:

- o canal recomendado (Celular ou Telefone Fixo);
- a confiança do modelo (probabilidade de conversão prevista para cada canal);
- se a decisão veio de exploração ou explotação, e por quê.

## 9. Ciclo de vida MLOps (Etapa 7)

O experimento é rastreado com **MLflow**:

- Experimento: `Datathon_Otimizacao_Canais`
- Run: `Epsilon_Greedy_RandomForest`
- Parâmetros registrados: `algoritmo_adaptativo`, `modelo_base`, `max_depth`,
  `n_estimators`, `epsilon`
- Métricas registradas: `taxa_baseline_fixa`, `taxa_epsilon_greedy`
- Modelo treinado (`modelo_contexto`) versionado via `mlflow.sklearn.log_model`

Para visualizar os runs localmente:

```bash
mlflow ui
```

## 10. Arquitetura-alvo em nuvem (AWS)

Para levar essa solução a produção, usaríamos a AWS como provedor de nuvem. O
**pipeline de dados e treino** ficaria assim: os dados brutos e as versões
processadas do dataset seriam armazenados no **S3**, servindo tanto de fonte para
o treinamento quanto de artifact store do MLflow (rodando em uma instância **EC2**
ou containerizado no **ECS/Fargate**, com o backend de metadados em um banco
**RDS PostgreSQL**). O treinamento e a re-execução periódica do modelo de contexto
seriam orquestrados via **SageMaker Training Jobs** ou uma função agendada no
**EventBridge + Step Functions**, permitindo retreinar o modelo à medida que novos
dados de conversão chegam.

Para o **serving**, o modelo treinado seria publicado como um **endpoint do
SageMaker** (ou, alternativamente, empacotado como container na **ECR** e exposto
via **API Gateway + Lambda** para chamadas de baixa latência, reaproveitando a
função `recomendar_canal`). O **CloudWatch** cuidaria de logs, métricas de latência
e alarmes de drift/erro do modelo, o **IAM** controlaria o acesso aos dados e ao
endpoint (mantendo decisões sensíveis com humano no loop quando necessário), e o
**S3 + Glue/Athena** permitiriam consultar o histórico de decisões para auditorias
e novas avaliações offline do algoritmo adaptativo.

## 11. Checklist do desafio

- [x] Repositório organizado com código e dependências
- [x] Notebook de EDA com a base Kaggle limpa e referenciada
- [x] Modelo Baseline e Modelo Adaptativo implementados e comparados
- [x] Golden Set com 5 casos de teste
- [x] Código executável retornando a recomendação (`recomendar_canal`)
- [x] README preenchido (link da base, infraestrutura AWS, instruções de execução)
- [x] Tracking de experimentos via MLflow
- [ ] Vídeo de apresentação (até 5 min)