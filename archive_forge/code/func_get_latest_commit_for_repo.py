import functools
import logging
import os
from typing import Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
@functools.lru_cache(maxsize=1)
def get_latest_commit_for_repo(repo: str) -> str:
    """
    Fetches the latest commit hash for a repository from the HuggingFace model hub.
    """
    try:
        import huggingface_hub as hub
    except ImportError:
        raise MlflowException('Unable to fetch model commit hash from the HuggingFace model hub. This is required for saving Transformer model without base model weights, while ensuring the version consistency of the model. Please install the `huggingface-hub` package and retry.', error_code=RESOURCE_DOES_NOT_EXIST)
    return hub.HfApi().model_info(repo).sha