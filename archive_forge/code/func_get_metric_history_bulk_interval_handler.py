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
def get_metric_history_bulk_interval_handler():
    MAX_RUNS_GET_METRIC_HISTORY_BULK = 100
    MAX_RESULTS_PER_RUN = 2500
    MAX_RESULTS_GET_METRIC_HISTORY = 25000
    request_message = _get_request_message(GetMetricHistoryBulkInterval(), schema={'run_ids': [_assert_required, _assert_array, _assert_item_type_string, lambda x: _assert_less_than_or_equal(len(x), MAX_RUNS_GET_METRIC_HISTORY_BULK, message=f'GetMetricHistoryBulkInterval request must specify at most {MAX_RUNS_GET_METRIC_HISTORY_BULK} run_ids. Received {len(x)} run_ids.')], 'metric_key': [_assert_required, _assert_string], 'start_step': [_assert_intlike], 'end_step': [_assert_intlike], 'max_results': [_assert_intlike, lambda x: _assert_intlike_within_range(int(x), 1, MAX_RESULTS_PER_RUN, message=f'max_results must be between 1 and {MAX_RESULTS_PER_RUN}.')]})
    args = request.args
    run_ids = request_message.run_ids
    metric_key = request_message.metric_key
    max_results = int(args.get('max_results', MAX_RESULTS_PER_RUN))
    store = _get_tracking_store()

    def _get_sampled_steps(run_ids, metric_key, max_results):
        start_step = args.get('start_step')
        end_step = args.get('end_step')
        if start_step is not None and end_step is not None:
            start_step = int(start_step)
            end_step = int(end_step)
            if start_step > end_step:
                raise MlflowException.invalid_parameter_value(f'end_step must be greater than start_step. Found start_step={start_step} and end_step={end_step}.')
        elif start_step is not None or end_step is not None:
            raise MlflowException.invalid_parameter_value('If either start step or end step are specified, both must be specified.')
        all_runs = [[m.step for m in store.get_metric_history(run_id, metric_key)] for run_id in run_ids]
        all_mins_and_maxes = {step for run in all_runs if run for step in [min(run), max(run)]}
        all_steps = sorted({step for sublist in all_runs for step in sublist})
        if start_step is None and end_step is None:
            start_step = 0
            end_step = all_steps[-1] if all_steps else 0
        all_mins_and_maxes = {step for step in all_mins_and_maxes if start_step <= step <= end_step}
        sampled_steps = _get_sampled_steps_from_steps(start_step, end_step, max_results, all_steps)
        return sorted(sampled_steps.union(all_mins_and_maxes))

    def _default_history_bulk_interval_impl():
        steps = _get_sampled_steps(run_ids, metric_key, max_results)
        metrics_with_run_ids = []
        for run_id in run_ids:
            metrics_with_run_ids.extend(store.get_metric_history_bulk_interval_from_steps(run_id=run_id, metric_key=metric_key, steps=steps, max_results=MAX_RESULTS_GET_METRIC_HISTORY))
        return metrics_with_run_ids
    metrics_with_run_ids = _default_history_bulk_interval_impl()
    response_message = GetMetricHistoryBulkInterval.Response()
    response_message.metrics.extend([m.to_proto() for m in metrics_with_run_ids])
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response