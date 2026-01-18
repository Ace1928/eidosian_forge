import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def get_conda_command(conda_env_name):
    if not is_windows() and (CONDA_EXE in os.environ or MLFLOW_CONDA_HOME.defined):
        conda_path = get_conda_bin_executable('conda')
        activate_conda_env = [f'source {os.path.dirname(conda_path)}/../etc/profile.d/conda.sh']
        activate_conda_env += [f'conda activate {conda_env_name} 1>&2']
    else:
        activate_path = get_conda_bin_executable('activate')
        if not is_windows():
            return [f'source {activate_path} {conda_env_name} 1>&2']
        else:
            return [f'conda activate {conda_env_name}']
    return activate_conda_env