import json
import time
from typing import AsyncIterable
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import AnthropicConfig, RouteConfig
from mlflow.gateway.constants import (
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions
@classmethod
def model_to_chat(cls, resp, config):
    stop_reason = 'length' if resp['stop_reason'] == 'max_tokens' else 'stop'
    return chat.ResponsePayload(id=resp['id'], created=int(time.time()), object='chat.completion', model=resp['model'], choices=[chat.Choice(index=0, message=chat.ResponseMessage(role='assistant', content=c['text']), finish_reason=stop_reason) for c in resp['content']], usage=chat.ChatUsage(prompt_tokens=resp['usage']['input_tokens'], completion_tokens=resp['usage']['output_tokens'], total_tokens=resp['usage']['input_tokens'] + resp['usage']['output_tokens']))