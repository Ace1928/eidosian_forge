import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def error_rate(true_label, predicted_positive_class_proba):
    if true_label == positive_class:
        return 1 - predicted_positive_class_proba
    else:
        return predicted_positive_class_proba