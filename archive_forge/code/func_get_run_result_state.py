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
def get_run_result_state(self, databricks_run_id):
    """
        Get the run result state (string) of a Databricks job run.

        Args:
            databricks_run_id: Integer Databricks job run ID.

        Returns:
            `RunResultState <https://docs.databricks.com/api/latest/jobs.html#runresultstate>`_ or
            None if the run is still active.
        """
    res = self.jobs_runs_get(databricks_run_id)
    return res['state'].get('result_state', None)