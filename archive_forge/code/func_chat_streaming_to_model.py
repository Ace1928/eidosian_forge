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
def chat_streaming_to_model(cls, payload, config):
    return cls.chat_to_model(payload, config)