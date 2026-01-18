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
@_use_repl_context_if_available('isInCluster')
def is_in_cluster():
    try:
        spark_session = _get_active_spark_session()
        return spark_session is not None and spark_session.conf.get('spark.databricks.clusterUsageTags.clusterId', None) is not None
    except Exception:
        return False