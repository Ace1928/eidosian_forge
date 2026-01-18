import time
from typing import Any, Dict
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import MistralConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import completions, embeddings
@classmethod
def model_to_embeddings(cls, resp, config):
    return embeddings.ResponsePayload(data=[embeddings.EmbeddingObject(embedding=data['embedding'], index=data['index']) for data in resp['data']], model=config.model.name, usage=embeddings.EmbeddingsUsage(prompt_tokens=resp['usage']['prompt_tokens'], total_tokens=resp['usage']['total_tokens']))