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
def get_databricks_model_version_url(registry_uri: str, name: str, version: str) -> Optional[str]:
    """Obtains a Databricks URL corresponding to the specified Model Version.

    Args:
        tracking_uri: The URI of the Model Registry server containing the Model Version.
        name: The name of the registered model containing the Model Version.
        version: Version number of the Model Version.

    Returns:
        A Databricks URL corresponding to the specified Model Version, or None if the
        Model Version does not belong to a Databricks Workspace.

    """
    try:
        workspace_info = DatabricksWorkspaceInfo.from_environment() or get_databricks_workspace_info_from_uri(registry_uri)
        if workspace_info is not None:
            return _construct_databricks_model_version_url(host=workspace_info.host, name=name, version=version, workspace_id=workspace_info.workspace_id)
    except Exception:
        return None