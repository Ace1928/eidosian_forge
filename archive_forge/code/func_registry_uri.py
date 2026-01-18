import os
from typing import Any, Dict, Optional, Union
import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
@property
def registry_uri(self) -> Optional[str]:
    return self.client._registry_uri