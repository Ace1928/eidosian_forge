import logging
import os
import time
from datetime import datetime
from shlex import quote, split
from threading import RLock
import kubernetes
from kubernetes.config.config_exception import ConfigException
import docker
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
def push_image_to_registry(image_tag):
    client = docker.from_env(timeout=_DOCKER_API_TIMEOUT)
    _logger.info('=== Pushing docker image %s ===', image_tag)
    for line in client.images.push(repository=image_tag, stream=True, decode=True):
        if 'error' in line and line['error']:
            raise ExecutionException('Error while pushing to docker registry: {error}'.format(error=line['error']))
    return client.images.get_registry_data(image_tag).id