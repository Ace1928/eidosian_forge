import json
import logging
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional
import mlflow
from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.utils import get_recipe_name
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
def get_run_tags_env_vars(recipe_root_path: str) -> Dict[str, str]:
    """
    Returns environment variables that should be set during step execution to ensure that MLflow
    Run Tags from the current context are applied to any MLflow Runs that are created during
    recipe execution.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        A dictionary of environment variable names and values.
    """
    run_context_tags = resolve_tags()
    git_tags = {}
    git_repo_url = get_git_repo_url(path=recipe_root_path)
    if git_repo_url:
        git_tags[MLFLOW_SOURCE_NAME] = git_repo_url
        git_tags[MLFLOW_GIT_REPO_URL] = git_repo_url
        git_tags[LEGACY_MLFLOW_GIT_REPO_URL] = git_repo_url
    git_commit = get_git_commit(path=recipe_root_path)
    if git_commit:
        git_tags[MLFLOW_GIT_COMMIT] = git_commit
    git_branch = get_git_branch(path=recipe_root_path)
    if git_branch:
        git_tags[MLFLOW_GIT_BRANCH] = git_branch
    return {MLFLOW_RUN_CONTEXT.name: json.dumps({**run_context_tags, **git_tags})}