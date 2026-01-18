import datetime
import logging
import operator
import os
import sys
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.metrics import (
from mlflow.recipes.utils.step import get_merged_eval_metrics, validate_classification_config
from mlflow.recipes.utils.tracking import (
from mlflow.tracking.fluent import _get_experiment_id, _set_experiment_primary_metric
from mlflow.utils.databricks_utils import get_databricks_env_vars, get_databricks_run_url
from mlflow.utils.string_utils import strip_prefix
def get_icon(validated):
    return '✅' if validated else '❌'