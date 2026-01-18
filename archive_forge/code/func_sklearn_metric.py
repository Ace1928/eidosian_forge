import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple
import pandas as pd
import mlflow
from mlflow import MlflowException
from mlflow.models import EvaluationMetric
from mlflow.models.evaluation.default_evaluator import (
from mlflow.recipes.utils.metrics import RecipeMetric, _load_custom_metrics
def sklearn_metric(X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args):
    custom_metrics_mod = importlib.import_module('sklearn.metrics')
    eval_fn = getattr(custom_metrics_mod, metric_name)
    val_metric = coeff * eval_fn(y_val, estimator.predict(X_val), average=avg)
    train_metric = coeff * eval_fn(y_train, estimator.predict(X_train), average=avg)
    return (val_metric, {f'{metric_name}_train': train_metric, f'{metric_name}_val': val_metric})