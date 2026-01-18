from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple
from fastapi import HTTPException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.schemas import chat, completions, embeddings
@staticmethod
def check_for_model_field(payload):
    from fastapi import HTTPException
    if 'model' in payload:
        raise HTTPException(status_code=422, detail="The parameter 'model' is not permitted to be passed. The route being queried already defines a model instance.")