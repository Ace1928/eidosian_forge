import json
import logging
import os
import platform
import signal
import sys
import tarfile
import time
import urllib.parse
from subprocess import Popen
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.container import (
from mlflow.models.container import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_unique_resource_id
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data
def status_check_fn():
    if time.time() - operation_start_time < 20:
        return _SageMakerOperationStatus.in_progress()
    endpoint_info = sage_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_update_was_rolled_back = endpoint_info['EndpointStatus'] == 'InService' and endpoint_info['EndpointConfigName'] != new_config_name
    if endpoint_update_was_rolled_back or endpoint_info['EndpointStatus'] == 'Failed':
        failure_reason = endpoint_info.get('FailureReason', 'An unknown SageMaker failure occurred. Please see the SageMaker console logs for more information.')
        return _SageMakerOperationStatus.failed(failure_reason)
    elif endpoint_info['EndpointStatus'] == 'InService':
        return _SageMakerOperationStatus.succeeded('The SageMaker endpoint was updated successfully.')
    else:
        return _SageMakerOperationStatus.in_progress('The update operation is still in progress. Current endpoint status: "{endpoint_status}"'.format(endpoint_status=endpoint_info['EndpointStatus']))