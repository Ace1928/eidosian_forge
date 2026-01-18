import ctypes
import logging
import os
import pathlib
import posixpath
import shlex
import signal
import subprocess
import sys
import warnings
from pathlib import Path
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend, docker_utils
from mlflow.models.docker_utils import PYTHON_SLIM_BASE_IMAGE, UBUNTU_BASE_IMAGE
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import (
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_conda_bin_executable, get_or_create_conda_env
from mlflow.utils.environment import Environment, _PythonEnv
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import _get_all_flavor_configurations
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.os import is_windows
from mlflow.utils.process import ShellCommandException, cache_return_value_per_process
from mlflow.utils.virtualenv import (
from mlflow.version import VERSION
def serve_stdin(self, model_uri, stdout=None, stderr=None):
    local_path = _download_artifact_from_uri(model_uri)
    return self.prepare_env(local_path).execute(command=f'python {_STDIN_SERVER_SCRIPT} --model-uri {local_path}', stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, synchronous=False)