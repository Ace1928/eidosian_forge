import logging
import os
import pathlib
import posixpath
from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml
def get_default_profile() -> str:
    """
    Returns the default profile name under which a recipe is executed. The default
    profile may change depending on runtime environment.

    Returns:
        The default profile name string.
    """
    return 'databricks' if is_in_databricks_runtime() else 'local'