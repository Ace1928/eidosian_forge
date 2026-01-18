import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def get_or_create_conda_env(conda_env_path, env_id=None, capture_output=False, env_root_dir=None, pip_requirements_override=None):
    """Given a `Project`, creates a conda environment containing the project's dependencies if such
    a conda environment doesn't already exist. Returns the name of the conda environment.

    Args:
        conda_env_path: Path to a conda yaml file.
        env_id: Optional string that is added to the contents of the yaml file before
            calculating the hash. It can be used to distinguish environments that have the
            same conda dependencies but are supposed to be different based on the context.
            For example, when serving the model we may install additional dependencies to the
            environment after the environment has been activated.
        capture_output: Specify the capture_output argument while executing the
            "conda env create" command.
        env_root_dir: See doc of PyFuncBackend constructor argument `env_root_dir`.
        pip_requirements_override: If specified, install the specified python dependencies to
            the environment (upgrade if already installed).

    Returns:
        The name of the conda environment.

    """
    conda_path = get_conda_bin_executable('conda')
    conda_env_create_path = _get_conda_executable_for_create_env()
    try:
        process._exec_cmd([conda_path, '--help'], throw_on_error=False)
    except OSError:
        raise ExecutionException(f'Could not find Conda executable at {conda_path}. Ensure Conda is installed as per the instructions at https://conda.io/projects/conda/en/latest/user-guide/install/index.html. You can also configure MLflow to look for a specific Conda executable by setting the {MLFLOW_CONDA_HOME} environment variable to the path of the Conda executable')
    try:
        process._exec_cmd([conda_env_create_path, '--help'], throw_on_error=False)
    except OSError:
        raise ExecutionException(f'You have set the env variable {MLFLOW_CONDA_CREATE_ENV_CMD}, but {conda_env_create_path} does not exist or it is not working properly. Note that {conda_env_create_path} and the conda executable need to be in the same conda environment. You can change the search path bymodifying the env variable {MLFLOW_CONDA_HOME}')
    conda_extra_env_vars = _get_conda_extra_env_vars(env_root_dir)
    project_env_name = _get_conda_env_name(conda_env_path, env_id=env_id, env_root_dir=env_root_dir)
    if env_root_dir is not None:
        project_env_path = os.path.join(env_root_dir, _CONDA_ENVS_DIR, project_env_name)
    else:
        project_env_path = project_env_name
    if project_env_name in _list_conda_environments(conda_extra_env_vars):
        _logger.info('Conda environment %s already exists.', project_env_path)
        return Environment(get_conda_command(project_env_name), conda_extra_env_vars)
    _logger.info('=== Creating conda environment %s ===', project_env_path)
    try:
        _create_conda_env_func = _create_conda_env_retry if 'PYTEST_CURRENT_TEST' in os.environ else _create_conda_env
        conda_env = _create_conda_env_func(conda_env_path, conda_env_create_path, project_env_name, conda_extra_env_vars, capture_output)
        if pip_requirements_override:
            _logger.info(f'Installing additional dependencies specifiedby pip_requirements_override: {pip_requirements_override}')
            cmd = [conda_path, 'install', '-n', project_env_name, '--yes', '--quiet', *pip_requirements_override]
            process._exec_cmd(cmd, extra_env=conda_extra_env_vars, capture_output=capture_output)
        return conda_env
    except Exception:
        try:
            if project_env_name in _list_conda_environments(conda_extra_env_vars):
                _logger.warning('Encountered unexpected error while creating conda environment. Removing %s.', project_env_path)
                process._exec_cmd([conda_path, 'remove', '--yes', '--name', project_env_name, '--all'], extra_env=conda_extra_env_vars, capture_output=False)
        except Exception as e:
            _logger.warning('Removing conda environment %s failed (error: %s)', project_env_path, repr(e))
        raise