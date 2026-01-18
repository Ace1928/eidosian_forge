import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
@_use_repl_context_if_available('notebookPath')
def get_notebook_path():
    """Should only be called if is_in_databricks_notebook is true"""
    path = _get_property_from_spark_context('spark.databricks.notebook.path')
    if path is not None:
        return path
    try:
        return _get_command_context().notebookPath().get()
    except Exception:
        return _get_extra_context('notebook_path')