import datetime
import importlib
import logging
import os
import re
import shutil
import sys
import warnings
import cloudpickle
import yaml
import mlflow
from mlflow.entities import SourceType, ViewType
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.recipes.artifacts import (
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.metrics import (
from mlflow.recipes.utils.step import (
from mlflow.recipes.utils.tracking import (
from mlflow.recipes.utils.wrapped_recipe_model import WrappedRecipeModel
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.databricks_utils import get_databricks_env_vars, get_databricks_run_url
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.string_utils import strip_prefix
def inverse_label_encoder(predicted_output):
    if not label_encoder:
        return predicted_output
    return label_encoder.inverse_transform(predicted_output)