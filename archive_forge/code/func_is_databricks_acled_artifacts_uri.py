import os
import pathlib
import posixpath
import re
import urllib.parse
import uuid
from typing import Any, Tuple
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.os import is_windows
from mlflow.utils.validation import _validate_db_type_string
def is_databricks_acled_artifacts_uri(artifact_uri):
    _ACLED_ARTIFACT_URI = 'databricks/mlflow-tracking/'
    artifact_uri_path = extract_and_normalize_path(artifact_uri)
    return artifact_uri_path.startswith(_ACLED_ARTIFACT_URI)