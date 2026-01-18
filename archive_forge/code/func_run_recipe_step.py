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
def run_recipe_step(recipe_root_path: str, recipe_steps: List[BaseStep], target_step: BaseStep, template: str) -> BaseStep:
    """
    Runs the specified step in the specified recipe, as well as all dependent steps.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_steps: A list of all the steps contained in the subgraph of the specified
            recipe that contains the target_step. Recipe steps must be provided in the order
            that they are intended to be executed.
        target_step: The step to run.
        template: The template to use when selecting a Makefile to load. If the template is
            invalid, an exception is thrown.

    Returns:
        The last step that successfully completed during the recipe execution. If execution
        was successful, this always corresponds to the supplied target step. If execution was
        unsuccessful, this corresponds to the step that failed.
    """
    target_step_index = recipe_steps.index(target_step)
    execution_dir_path = _get_or_create_execution_directory(recipe_root_path, recipe_steps, template)

    def get_execution_state(step):
        return step.get_execution_state(output_directory=_get_step_output_directory_path(execution_directory_path=execution_dir_path, step_name=step.name))
    clean_execution_state(recipe_root_path=recipe_root_path, recipe_steps=[step for step in recipe_steps[:target_step_index + 1] if get_execution_state(step).status != StepStatus.SUCCEEDED])
    _write_updated_step_confs(recipe_steps=recipe_steps, execution_directory_path=execution_dir_path)
    make_env = {MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME.name: target_step.name}
    for step in recipe_steps:
        make_env.update(step.environment)
    _run_make(execution_directory_path=execution_dir_path, rule_name=target_step.name, extra_env=make_env, recipe_steps=recipe_steps)
    last_executed_step = recipe_steps[0]
    last_executed_step_state = get_execution_state(last_executed_step)
    for step in recipe_steps[1:target_step_index + 1]:
        step_state = get_execution_state(step)
        if step_state.last_updated_timestamp >= last_executed_step_state.last_updated_timestamp:
            last_executed_step = step
            last_executed_step_state = step_state
    clean_execution_state(recipe_root_path=recipe_root_path, recipe_steps=[step for step in recipe_steps[recipe_steps.index(last_executed_step):] if get_execution_state(step).last_updated_timestamp < last_executed_step_state.last_updated_timestamp])
    return last_executed_step