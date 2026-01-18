import contextlib
import json
import logging
import os
import posixpath
import re
import sys
import tempfile
import urllib
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
import yaml
import mlflow
from mlflow.entities import DatasetInput, Experiment, FileInfo, Metric, Param, Run, RunTag, ViewType
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.entities.model_registry.model_version_stages import ALL_STAGES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.utils.models import (
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.client import ModelRegistryClient
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.tracking.artifact_utils import _upload_artifacts_to_databricks
from mlflow.tracking.multimedia import Image, compress_image_size, convert_to_pil_image
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.annotations import deprecated, experimental
from mlflow.utils.async_logging.run_operations import RunOperations
from mlflow.utils.databricks_utils import get_databricks_run_url
from mlflow.utils.logging_utils import eprint
from mlflow.utils.mlflow_tags import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import is_databricks_unity_catalog_uri, is_databricks_uri
from mlflow.utils.validation import (
def get_artifact_data(run):
    run_id = run.run_id
    norm_path = posixpath.normpath(artifact_file)
    artifact_dir = posixpath.dirname(norm_path)
    artifact_dir = None if artifact_dir == '' else artifact_dir
    existing_predictions = pd.DataFrame()
    artifacts = [f.path for f in self.list_artifacts(run_id, path=artifact_dir) if not f.is_dir]
    if artifact_file in artifacts:
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded_artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file, dst_path=tmpdir)
            existing_predictions = pd.read_json(downloaded_artifact_path, orient='split')
            if extra_columns is not None:
                for column in extra_columns:
                    if column in existing_predictions:
                        column_name = f'{column}_'
                        _logger.warning(f'Column name {column} already exists in the table. Resolving the conflict, by appending an underscore to the column name.')
                    else:
                        column_name = column
                    existing_predictions[column_name] = run[column]
    else:
        raise MlflowException(f'Artifact {artifact_file} not found for run {run_id}.', RESOURCE_DOES_NOT_EXIST)
    return existing_predictions