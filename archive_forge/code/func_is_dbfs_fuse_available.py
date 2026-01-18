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
def is_dbfs_fuse_available():
    with open(os.devnull, 'w') as devnull_stderr, open(os.devnull, 'w') as devnull_stdout:
        try:
            return subprocess.call(['mountpoint', '/dbfs'], stderr=devnull_stderr, stdout=devnull_stdout) == 0
        except Exception:
            return False