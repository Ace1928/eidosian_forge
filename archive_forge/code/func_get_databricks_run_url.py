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
def get_databricks_run_url(tracking_uri: str, run_id: str, artifact_path=None) -> Optional[str]:
    """
    Obtains a Databricks URL corresponding to the specified MLflow Run, optionally referring
    to an artifact within the run.

    Args:
        tracking_uri: The URI of the MLflow Tracking server containing the Run.
        run_id: The ID of the MLflow Run for which to obtain a Databricks URL.
        artifact_path: An optional relative artifact path within the Run to which the URL
            should refer.

    Returns:
        A Databricks URL corresponding to the specified MLflow Run
        (and artifact path, if specified), or None if the MLflow Run does not belong to a
        Databricks Workspace.
    """
    from mlflow.tracking.client import MlflowClient
    try:
        workspace_info = DatabricksWorkspaceInfo.from_environment() or get_databricks_workspace_info_from_uri(tracking_uri)
        if workspace_info is not None:
            experiment_id = MlflowClient(tracking_uri).get_run(run_id).info.experiment_id
            return _construct_databricks_run_url(host=workspace_info.host, experiment_id=experiment_id, run_id=run_id, workspace_id=workspace_info.workspace_id, artifact_path=artifact_path)
    except Exception:
        return None