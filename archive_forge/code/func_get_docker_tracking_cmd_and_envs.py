import logging
import os
import posixpath
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import docker
from mlflow import tracking
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException
from mlflow.projects.utils import MLFLOW_DOCKER_WORKDIR_PATH
from mlflow.utils import file_utils, process
from mlflow.utils.databricks_utils import get_databricks_env_vars
from mlflow.utils.file_utils import _handle_readonly_on_windows
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_DOCKER_IMAGE_ID, MLFLOW_DOCKER_IMAGE_URI
def get_docker_tracking_cmd_and_envs(tracking_uri):
    cmds = []
    env_vars = {}
    local_path, container_tracking_uri = _get_local_uri_or_none(tracking_uri)
    if local_path is not None:
        cmds = ['-v', f'{local_path}:{_MLFLOW_DOCKER_TRACKING_DIR_PATH}']
        env_vars[MLFLOW_TRACKING_URI.name] = container_tracking_uri
    env_vars.update(get_databricks_env_vars(tracking_uri))
    return (cmds, env_vars)