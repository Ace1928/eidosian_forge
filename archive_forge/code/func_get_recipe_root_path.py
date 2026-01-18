import logging
import os
import pathlib
import posixpath
from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml
def get_recipe_root_path() -> str:
    """
    Obtains the path of the recipe corresponding to the current working directory, throwing an
    ``MlflowException`` if the current working directory does not reside within a recipe
    directory.

    Returns:
        The absolute path of the recipe root directory on the local filesystem.
    """
    curr_dir_path = pathlib.Path.cwd()
    while True:
        recipe_yaml_path_to_check = curr_dir_path / _RECIPE_CONFIG_FILE_NAME
        if recipe_yaml_path_to_check.exists():
            return str(curr_dir_path.resolve())
        elif curr_dir_path != curr_dir_path.parent:
            curr_dir_path = curr_dir_path.parent
        else:
            raise MlflowException(f'Failed to find {_RECIPE_CONFIG_FILE_NAME}!')