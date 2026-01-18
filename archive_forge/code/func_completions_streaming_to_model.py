from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple
from fastapi import HTTPException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.schemas import chat, completions, embeddings
@classmethod
def completions_streaming_to_model(cls, payload, config):
    raise NotImplementedError