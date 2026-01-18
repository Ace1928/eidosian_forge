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
def get_mlflow_credential_context_by_run_id(run_id):
    from mlflow.tracking.artifact_utils import get_artifact_uri
    from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
    run_root_artifact_uri = get_artifact_uri(run_id=run_id)
    profile = get_databricks_profile_uri_from_artifact_uri(run_root_artifact_uri)
    return MlflowCredentialContext(profile)