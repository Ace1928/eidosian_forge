import inspect
import json
import logging
import os
import shlex
import sys
import traceback
from typing import Dict, NamedTuple, Optional, Tuple
import flask
from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import _log_warning_if_params_not_in_predict_signature
from mlflow.types import ParamSchema, Schema
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.proto_json_utils import (
from mlflow.version import VERSION
from io import StringIO
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.server.handlers import catch_mlflow_exception
def parse_csv_input(csv_input, schema: Schema=None):
    """
    Args:
        csv_input: A CSV-formatted string representation of a Pandas DataFrame, or a stream
                   containing such a string representation.
        schema: Optional schema specification to be used during parsing.
    """
    import pandas as pd
    try:
        if schema is None:
            return pd.read_csv(csv_input)
        else:
            dtypes = dict(zip(schema.input_names(), schema.pandas_types()))
            return pd.read_csv(csv_input, dtype=dtypes)
    except Exception as e:
        _handle_serving_error(error_message=f"Failed to parse input as a Pandas DataFrame. Ensure that the input is a valid CSV-formatted Pandas DataFrame produced using the `pandas.DataFrame.to_csv()` method. Error: '{e}'", error_code=BAD_REQUEST)