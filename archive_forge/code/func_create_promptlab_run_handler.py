import bisect
import json
import logging
import os
import pathlib
import posixpath
import re
import tempfile
import time
import urllib
from functools import wraps
from typing import List, Set
import requests
from flask import Response, current_app, jsonify, request, send_file
from google.protobuf import descriptor
from google.protobuf.json_format import ParseError
from mlflow.entities import DatasetInput, ExperimentTag, FileInfo, Metric, Param, RunTag, ViewType
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.models import Model
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server.validation import _validate_content_type
from mlflow.store.artifact.artifact_repo import MultipartUploadMixin
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.promptlab_utils import _create_promptlab_run_impl
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import is_local_uri, validate_path_is_safe, validate_query_string
from mlflow.utils.validation import _validate_batch_log_api_req
@catch_mlflow_exception
@_disable_if_artifacts_only
def create_promptlab_run_handler():

    def assert_arg_exists(arg_name, arg):
        if not arg:
            raise MlflowException(message=f'CreatePromptlabRun request must specify {arg_name}.', error_code=INVALID_PARAMETER_VALUE)
    _validate_content_type(request, ['application/json'])
    args = request.json
    experiment_id = args.get('experiment_id')
    assert_arg_exists('experiment_id', experiment_id)
    run_name = args.get('run_name', None)
    tags = args.get('tags', [])
    prompt_template = args.get('prompt_template')
    assert_arg_exists('prompt_template', prompt_template)
    raw_prompt_parameters = args.get('prompt_parameters')
    assert_arg_exists('prompt_parameters', raw_prompt_parameters)
    prompt_parameters = [Param(param.get('key'), param.get('value')) for param in args.get('prompt_parameters')]
    model_route = args.get('model_route')
    assert_arg_exists('model_route', model_route)
    raw_model_parameters = args.get('model_parameters', [])
    model_parameters = [Param(param.get('key'), param.get('value')) for param in raw_model_parameters]
    model_input = args.get('model_input')
    assert_arg_exists('model_input', model_input)
    model_output = args.get('model_output', None)
    raw_model_output_parameters = args.get('model_output_parameters', [])
    model_output_parameters = [Param(param.get('key'), param.get('value')) for param in raw_model_output_parameters]
    mlflow_version = args.get('mlflow_version')
    assert_arg_exists('mlflow_version', mlflow_version)
    user_id = args.get('user_id', 'unknown')
    start_time = args.get('start_time', int(time.time() * 1000))
    store = _get_tracking_store()
    run = _create_promptlab_run_impl(store, experiment_id=experiment_id, run_name=run_name, tags=tags, prompt_template=prompt_template, prompt_parameters=prompt_parameters, model_route=model_route, model_parameters=model_parameters, model_input=model_input, model_output=model_output, model_output_parameters=model_output_parameters, mlflow_version=mlflow_version, user_id=user_id, start_time=start_time)
    response_message = CreateRun.Response()
    response_message.run.MergeFrom(run.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response