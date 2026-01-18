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
def cleanup_fn():
    _logger.info('Cleaning up unused resources...')
    if mode == DEPLOYMENT_MODE_REPLACE:
        for pv in deployed_production_variants:
            deployed_model_arn = _delete_sagemaker_model(model_name=pv['ModelName'], sage_client=sage_client, s3_client=s3_client)
            _logger.info('Deleted model with arn: %s', deployed_model_arn)
    sage_client.delete_endpoint_config(EndpointConfigName=deployed_config_name)
    _logger.info('Deleted endpoint configuration with arn: %s', deployed_config_arn)