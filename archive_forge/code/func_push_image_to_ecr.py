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
def push_image_to_ecr(image=DEFAULT_IMAGE_NAME):
    """
    Push local Docker image to AWS ECR.

    The image is pushed under currently active AWS account and to the currently active AWS region.

    Args:
        image: Docker image name.
    """
    import boto3
    _logger.info('Pushing image to ECR')
    client = boto3.client('sts')
    caller_id = client.get_caller_identity()
    account = caller_id['Account']
    my_session = boto3.session.Session()
    region = my_session.region_name or 'us-west-2'
    fullname = _full_template.format(account=account, region=region, image=image, version=mlflow.version.VERSION)
    _logger.info('Pushing docker image %s to %s', image, fullname)
    ecr_client = boto3.client('ecr')
    try:
        ecr_client.describe_repositories(repositoryNames=[image])['repositories']
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=image)
        _logger.info('Created new ECR repository: %s', image)
    docker_login_cmd = f'aws ecr get-login-password | docker login  --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com'
    os_command_separator = ';\n'
    if platform.system() == 'Windows':
        os_command_separator = ' && '
    docker_tag_cmd = f'docker tag {image} {fullname}'
    docker_push_cmd = f'docker push {fullname}'
    cmd = os_command_separator.join([docker_login_cmd, docker_tag_cmd, docker_push_cmd])
    _logger.info('Executing: %s', cmd)
    os.system(cmd)