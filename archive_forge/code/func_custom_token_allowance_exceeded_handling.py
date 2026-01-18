import time
from contextlib import contextmanager
from typing import Any, Dict, List
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import MosaicMLConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings
@contextmanager
def custom_token_allowance_exceeded_handling():
    """
    Context manager handler for specific error messages that are incorrectly set as server-side
    errors, but are in actuality an issue with the request sent to the external provider.
    """
    try:
        yield
    except HTTPException as e:
        status_code = e.status_code
        detail = e.detail or {}
        if status_code == 500 and detail and any((detail.get('message', '').startswith(x) for x in ('Error: max output tokens is limited to', 'Error: prompt token count'))):
            raise HTTPException(status_code=422, detail=detail)
        else:
            raise