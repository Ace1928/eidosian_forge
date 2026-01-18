import time
from typing import Any, Dict
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import MistralConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import completions, embeddings
@classmethod
def model_to_completions(cls, resp, config):
    return completions.ResponsePayload(created=int(time.time()), object='text_completion', model=config.model.name, choices=[completions.Choice(index=idx, text=c['message']['content'], finish_reason=c['finish_reason']) for idx, c in enumerate(resp['choices'])], usage=completions.CompletionsUsage(prompt_tokens=resp['usage']['prompt_tokens'], completion_tokens=resp['usage']['completion_tokens'], total_tokens=resp['usage']['total_tokens']))