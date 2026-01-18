import os
import platform
import shutil
import subprocess
import sys
import yaml
import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.pyfunc.model import MLMODEL_FILE_NAME, Model
from mlflow.store.artifact.utils.models import _parse_model_uri, get_model_name_and_version
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
@classmethod
def get_wheel_artifact_path(cls, original_artifact_path):
    return original_artifact_path + '_' + _WHEELS_FOLDER_NAME