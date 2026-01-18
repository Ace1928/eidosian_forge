import logging
import os
import shutil
import subprocess
import hashlib
import json
from typing import Optional, List, Union, Tuple
def create_conda_env_if_needed(conda_yaml_file: str, prefix: str, logger: Optional[logging.Logger]=None) -> None:
    """
    Given a conda YAML, creates a conda environment containing the required
    dependencies if such a conda environment doesn't already exist.
    Args:
        conda_yaml_file: The path to a conda `environment.yml` file.
        prefix: Directory to install the environment into via
            the `--prefix` option to conda create.  This also becomes the name
            of the conda env; i.e. it can be passed into `conda activate` and
            `conda remove`
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    conda_path = get_conda_bin_executable('conda')
    try:
        exec_cmd([conda_path, '--help'], throw_on_error=False)
    except (EnvironmentError, FileNotFoundError):
        raise ValueError(f"Could not find Conda executable at '{conda_path}'. Ensure Conda is installed as per the instructions at https://conda.io/projects/conda/en/latest/user-guide/install/index.html. You can also configure Ray to look for a specific Conda executable by setting the {RAY_CONDA_HOME} environment variable to the path of the Conda executable.")
    _, stdout, _ = exec_cmd([conda_path, 'env', 'list', '--json'])
    envs = json.loads(stdout)['envs']
    if prefix in envs:
        logger.info(f'Conda environment {prefix} already exists.')
        return
    create_cmd = [conda_path, 'env', 'create', '--file', conda_yaml_file, '--prefix', prefix]
    logger.info(f'Creating conda environment {prefix}')
    exit_code, output = exec_cmd_stream_to_logger(create_cmd, logger)
    if exit_code != 0:
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
        raise RuntimeError(f'Failed to install conda environment {prefix}:\nOutput:\n{output}')