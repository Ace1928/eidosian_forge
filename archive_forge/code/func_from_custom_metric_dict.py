import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
@classmethod
def from_custom_metric_dict(cls, custom_metric_dict):
    metric_name = custom_metric_dict.get(RecipeMetric._KEY_METRIC_NAME)
    greater_is_better = custom_metric_dict.get(RecipeMetric._KEY_METRIC_GREATER_IS_BETTER)
    custom_function = custom_metric_dict.get(RecipeMetric._KEY_CUSTOM_FUNCTION)
    if (metric_name, greater_is_better, custom_function).count(None) > 0:
        raise MlflowException(f'Invalid custom metric definition: {custom_metric_dict}', error_code=INVALID_PARAMETER_VALUE)
    return cls(name=metric_name, greater_is_better=greater_is_better, custom_function=custom_function)