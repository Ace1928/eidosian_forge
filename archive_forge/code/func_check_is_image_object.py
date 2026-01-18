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
def check_is_image_object(obj):
    return hasattr(obj, 'save') and callable(getattr(obj, 'save')) and hasattr(obj, 'resize') and callable(getattr(obj, 'resize')) and hasattr(obj, 'size')