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
def prepare_env(self, model_uri, capture_output=False, pip_requirements_override=None):
    if self._environment is not None:
        return self._environment

    @cache_return_value_per_process
    def _get_or_create_env_root_dir(should_use_nfs):
        if should_use_nfs:
            root_tmp_dir = get_or_create_nfs_tmp_dir()
        else:
            root_tmp_dir = get_or_create_tmp_dir()
        env_root_dir = os.path.join(root_tmp_dir, 'envs')
        os.makedirs(env_root_dir, exist_ok=True)
        return env_root_dir
    local_path = _download_artifact_from_uri(model_uri)
    if self._create_env_root_dir:
        if self._env_root_dir is not None:
            raise Exception('env_root_dir can not be set when create_env_root_dir=True')
        nfs_root_dir = get_nfs_cache_root_dir()
        env_root_dir = _get_or_create_env_root_dir(nfs_root_dir is not None)
    else:
        env_root_dir = self._env_root_dir
    if self._env_manager == _EnvManager.VIRTUALENV:
        activate_cmd = _get_or_create_virtualenv(local_path, self._env_id, env_root_dir=env_root_dir, capture_output=capture_output, pip_requirements_override=pip_requirements_override)
        self._environment = Environment(activate_cmd)
    elif self._env_manager == _EnvManager.CONDA:
        conda_env_path = os.path.join(local_path, _extract_conda_env(self._config[ENV]))
        self._environment = get_or_create_conda_env(conda_env_path, env_id=self._env_id, capture_output=capture_output, env_root_dir=env_root_dir, pip_requirements_override=pip_requirements_override)
    elif self._env_manager == _EnvManager.LOCAL:
        raise Exception('Prepare env should not be called with local env manager!')
    else:
        raise Exception(f"Unexpected env manager value '{self._env_manager}'")
    if self._install_mlflow:
        self._environment.execute(_get_pip_install_mlflow())
    else:
        self._environment.execute('python -c ""')
    return self._environment