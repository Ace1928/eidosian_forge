import hashlib
import logging
import os
import pathlib
import re
import shutil
from typing import Dict, List
from mlflow.environment_variables import (
from mlflow.recipes.step import BaseStep, StepStatus
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd
def clean_execution_state(recipe_root_path: str, recipe_steps: List[BaseStep]) -> None:
    """
    Removes all execution state for the specified recipe steps from the associated execution
    directory on the local filesystem. This method does *not* remove other execution results, such
    as content logged to MLflow Tracking.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_steps: The recipe steps for which to remove execution state.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)
    for step in recipe_steps:
        step_outputs_path = _get_step_output_directory_path(execution_directory_path=execution_dir_path, step_name=step.name)
        if os.path.exists(step_outputs_path):
            shutil.rmtree(step_outputs_path)
        os.makedirs(step_outputs_path)