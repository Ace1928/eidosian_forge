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
def run_databricks(self, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec, run_id, env_manager):
    tracking_uri = _get_tracking_uri_for_run()
    dbfs_fuse_uri = self._upload_project_to_dbfs(work_dir, experiment_id)
    env_vars = {MLFLOW_TRACKING_URI.name: tracking_uri, MLFLOW_EXPERIMENT_ID.name: experiment_id}
    _logger.info('=== Running entry point %s of project %s on Databricks ===', entry_point, uri)
    command = _get_databricks_run_cmd(dbfs_fuse_uri, run_id, entry_point, parameters, env_manager)
    return self._run_shell_command_job(uri, command, env_vars, cluster_spec)