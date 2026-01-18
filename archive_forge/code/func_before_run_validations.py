import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import textwrap
import time
from shlex import quote
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import databricks_utils, file_utils, rest_utils
from mlflow.utils.mlflow_tags import (
from mlflow.utils.uri import is_databricks_uri, is_http_uri
from mlflow.version import VERSION, is_release_version
def before_run_validations(tracking_uri, backend_config):
    """Validations to perform before running a project on Databricks."""
    if backend_config is None:
        raise ExecutionException('Backend spec must be provided when launching MLflow project runs on Databricks.')
    elif 'existing_cluster_id' in backend_config:
        raise MlflowException(message='MLflow Project runs on Databricks must provide a *new cluster* specification. Project execution against existing clusters is not currently supported. For more information, see https://mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-databricks', error_code=INVALID_PARAMETER_VALUE)
    if not is_databricks_uri(tracking_uri) and (not is_http_uri(tracking_uri)):
        raise ExecutionException("When running on Databricks, the MLflow tracking URI must be of the form 'databricks' or 'databricks://profile', or a remote HTTP URI accessible to both the current client and code running on Databricks. Got local tracking URI %s. Please specify a valid tracking URI via mlflow.set_tracking_uri or by setting the MLFLOW_TRACKING_URI environment variable." % tracking_uri)