import time
from typing import List
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError, validator
from mlflow.gateway.config import MlflowModelServingConfig, RouteConfig
from mlflow.gateway.constants import MLFLOW_SERVING_RESPONSE_KEY
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings
@validator('predictions', pre=True)
def extract_choices(cls, predictions):
    if isinstance(predictions, list) and (not predictions):
        raise ValueError('The input list is empty')
    if isinstance(predictions, dict):
        if 'choices' not in predictions and len(predictions) > 1:
            raise ValueError("The dict format is invalid for this route type. Ensure the served model returns a dict key containing 'choices'")
        if len(predictions) == 1:
            predictions = next(iter(predictions.values()))
        else:
            predictions = predictions.get('choices', predictions)
        if not predictions:
            raise ValueError('The input list is empty')
    return predictions