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
def invocations(data, content_type, model, input_schema):
    type_parts = list(map(str.strip, content_type.split(';')))
    mime_type = type_parts[0]
    parameter_value_pairs = type_parts[1:]
    parameter_values = {key: value for pair in parameter_value_pairs for key, _, value in [pair.partition('=')]}
    charset = parameter_values.get('charset', 'utf-8').lower()
    if charset != 'utf-8':
        return InvocationsResponse(response='The scoring server only supports UTF-8', status=415, mimetype='text/plain')
    unexpected_content_parameters = set(parameter_values.keys()).difference({'charset'})
    if unexpected_content_parameters:
        return InvocationsResponse(response=f'Unrecognized content type parameters: {', '.join(unexpected_content_parameters)}. {SCORING_PROTOCOL_CHANGE_INFO}', status=415, mimetype='text/plain')
    should_parse_as_unified_llm_input = False
    if mime_type == CONTENT_TYPE_CSV:
        csv_input = StringIO(data)
        data = parse_csv_input(csv_input=csv_input, schema=input_schema)
        params = None
    elif mime_type == CONTENT_TYPE_JSON:
        json_input = _decode_json_input(data)
        should_parse_as_unified_llm_input = any((x in json_input for x in SUPPORTED_LLM_FORMATS))
        if should_parse_as_unified_llm_input:
            if hasattr(model.metadata, 'get_params_schema'):
                params_schema = model.metadata.get_params_schema()
            else:
                params_schema = None
            data, params = _split_data_and_params_for_llm_input(json_input, params_schema)
        else:
            data, params = _split_data_and_params(data)
            data = infer_and_parse_data(data, input_schema)
    else:
        return InvocationsResponse(response=f"This predictor only supports the following content types: Types: {CONTENT_TYPES}. Got '{flask.request.content_type}'.", status=415, mimetype='text/plain')
    try:
        if inspect.signature(model.predict).parameters.get('params'):
            raw_predictions = model.predict(data, params=params)
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
            raw_predictions = model.predict(data)
    except MlflowException as e:
        if 'Failed to enforce schema' in e.message:
            _logger.warning('If using `instances` as input key, we internally convert the data type from `records` (List[Dict]) type to `list` (Dict[str, List]) type if the data is a pandas dataframe representation. This might cause schema changes. Please use `inputs` to avoid this converesion.\n')
        e.message = f"Failed to predict data '{data}'. \nError: {e.message}"
        raise e
    except Exception:
        raise MlflowException(message='Encountered an unexpected error while evaluating the model. Verify that the serialized input Dataframe is compatible with the model for inference.', error_code=BAD_REQUEST, stack_trace=traceback.format_exc())
    result = StringIO()
    if should_parse_as_unified_llm_input:
        unwrapped_predictions_to_json(raw_predictions, result)
    else:
        predictions_to_json(raw_predictions, result)
    return InvocationsResponse(response=result.getvalue(), status=200, mimetype='application/json')