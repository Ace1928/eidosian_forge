import time
from typing import Any, Dict
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import MistralConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import completions, embeddings
@classmethod
def embeddings_to_model(cls, payload, config):
    return payload