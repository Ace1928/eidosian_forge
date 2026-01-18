import json
import logging
import os
import shutil
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union
import yaml
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import (
def load_input_example(self, path: str):
    """
        Load the input example saved along a model. Returns None if there is no example metadata
        (i.e. the model was saved without example). Raises FileNotFoundError if there is model
        metadata but the example file is missing.

        Args:
            path: Path to the model directory.

        Returns:
            Input example (NumPy ndarray, SciPy csc_matrix, SciPy csr_matrix,
            pandas DataFrame, dict) or None if the model has no example.
        """
    from mlflow.models.utils import _read_example
    return _read_example(self, path)