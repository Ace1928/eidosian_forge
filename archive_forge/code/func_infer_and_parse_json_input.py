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
@deprecated('infer_and_parse_data', '2.6.0')
def infer_and_parse_json_input(json_input, schema: Schema=None):
    """
    Args:
        json_input: A JSON-formatted string representation of TF serving input or a Pandas
                    DataFrame, or a stream containing such a string representation.
        schema: Optional schema specification to be used during parsing.
    """
    if isinstance(json_input, dict):
        decoded_input = json_input
    else:
        try:
            decoded_input = json.loads(json_input)
        except json.decoder.JSONDecodeError as ex:
            raise MlflowException(message=f"Failed to parse input from JSON. Ensure that input is a valid JSON formatted string. Error: '{ex}'. Input: \n{json_input}\n", error_code=BAD_REQUEST)
    if isinstance(decoded_input, dict):
        format_keys = set(decoded_input.keys()).intersection(SUPPORTED_FORMATS)
        if len(format_keys) != 1:
            message = f'Received dictionary with input fields: {list(decoded_input.keys())}'
            raise MlflowException(message=f'{REQUIRED_INPUT_FORMAT}. {message}. {SCORING_PROTOCOL_CHANGE_INFO}', error_code=BAD_REQUEST)
        input_format = format_keys.pop()
        if input_format in (INSTANCES, INPUTS):
            return parse_tf_serving_input(decoded_input, schema=schema)
        elif input_format in (DF_SPLIT, DF_RECORDS):
            pandas_orient = input_format[10:]
            return dataframe_from_parsed_json(decoded_input[input_format], pandas_orient=pandas_orient, schema=schema)
    elif isinstance(decoded_input, list):
        message = 'Received a list'
        raise MlflowException(message=f'{REQUIRED_INPUT_FORMAT}. {message}. {SCORING_PROTOCOL_CHANGE_INFO}', error_code=BAD_REQUEST)
    else:
        message = f"Received unexpected input type '{type(decoded_input)}'"
        raise MlflowException(message=f'{REQUIRED_INPUT_FORMAT}. {message}.', error_code=BAD_REQUEST)