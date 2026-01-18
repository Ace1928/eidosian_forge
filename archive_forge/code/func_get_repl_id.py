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
@_use_repl_context_if_available('replId')
def get_repl_id():
    """
    Returns:
        The ID of the current Databricks Python REPL.
    """
    try:
        dbutils = _get_dbutils()
        repl_id = dbutils.entry_point.getReplId()
        if repl_id is not None:
            return repl_id
    except Exception:
        pass
    try:
        from pyspark import SparkContext
        repl_id = SparkContext.getOrCreate().getLocalProperty('spark.databricks.replId')
        if repl_id is not None:
            return repl_id
    except Exception:
        pass