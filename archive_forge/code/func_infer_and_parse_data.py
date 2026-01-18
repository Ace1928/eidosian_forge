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
def infer_and_parse_data(data, schema: Schema=None):
    """
    Args:
        data: A dictionary representation of TF serving input or a Pandas
            DataFrame, or a stream containing such a string representation.
        schema: Optional schema specification to be used during parsing.
    """
    format_keys = set(data.keys()).intersection(SUPPORTED_FORMATS)
    if len(format_keys) != 1:
        message = f'Received dictionary with input fields: {list(data.keys())}'
        raise MlflowException(message=f'{REQUIRED_INPUT_FORMAT}. {message}. {SCORING_PROTOCOL_CHANGE_INFO}', error_code=BAD_REQUEST)
    input_format = format_keys.pop()
    if input_format in (INSTANCES, INPUTS):
        return parse_tf_serving_input(data, schema=schema)
    if input_format in (DF_SPLIT, DF_RECORDS):
        pandas_orient = input_format[10:]
        return dataframe_from_parsed_json(data[input_format], pandas_orient=pandas_orient, schema=schema)