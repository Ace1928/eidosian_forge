from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def infer_signature_from_llm_inference_task(inference_task: str, signature: Optional[ModelSignature]=None) -> ModelSignature:
    """
    Infers the signature according to the MLflow inference task.
    Raises exception if a signature is given.
    """
    inferred_signature = _SIGNATURE_FOR_LLM_INFERENCE_TASK[inference_task]
    if signature is not None and signature != inferred_signature:
        raise MlflowException(f'When `task` is specified as `{inference_task}`, the signature would be set by MLflow. Please do not set the signature.')
    return inferred_signature