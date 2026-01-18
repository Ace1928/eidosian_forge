import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
def make_metric(*, eval_fn, greater_is_better, name=None, long_name=None, version=None, metric_details=None):
    '''
    A factory function to create an :py:class:`EvaluationMetric` object.

    Args:
        eval_fn: A function that computes the metric with the following signature:

            .. code-block:: python

                def eval_fn(
                    predictions: pandas.Series,
                    targets: pandas.Series,
                    metrics: Dict[str, MetricValue],
                    **kwargs,
                ) -> Union[float, MetricValue]:
                    """
                    Args:
                        predictions: A pandas Series containing the predictions made by the model.
                        targets: (Optional) A pandas Series containing the corresponding labels
                            for the predictions made on that input.
                        metrics: (Optional) A dictionary containing the metrics calculated by the
                            default evaluator.  The keys are the names of the metrics and the values
                            are the metric values.  To access the MetricValue for the metrics
                            calculated by the system, make sure to specify the type hint for this
                            parameter as Dict[str, MetricValue].  Refer to the DefaultEvaluator
                            behavior section for what metrics will be returned based on the type of
                            model (i.e. classifier or regressor).  kwargs: Includes a list of args
                            that are used to compute the metric. These args could information coming
                            from input data, model outputs or parameters specified in the
                            `evaluator_config` argument of the `mlflow.evaluate` API.
                        kwargs: Includes a list of args that are used to compute the metric. These
                            args could be information coming from input data, model outputs,
                            other metrics, or parameters specified in the `evaluator_config`
                            argument of the `mlflow.evaluate` API.

                    Returns: MetricValue with per-row scores, per-row justifications, and aggregate
                        results.
                    """
                    ...

        greater_is_better: Whether a higher value of the metric is better.
        name: The name of the metric. This argument must be specified if ``eval_fn`` is a lambda
                    function or the ``eval_fn.__name__`` attribute is not available.
        long_name: (Optional) The long name of the metric. For example, ``"mean_squared_error"``
            for ``"mse"``.
        version: (Optional) The metric version. For example ``v1``.
        metric_details: (Optional) A description of the metric and how it is calculated.

    .. seealso::

        - :py:class:`mlflow.models.EvaluationMetric`
        - :py:func:`mlflow.evaluate`
    '''
    if name is None:
        if isinstance(eval_fn, FunctionType) and eval_fn.__name__ == '<lambda>':
            raise MlflowException('`name` must be specified if `eval_fn` is a lambda function.', INVALID_PARAMETER_VALUE)
        if not hasattr(eval_fn, '__name__'):
            raise MlflowException('`name` must be specified if `eval_fn` does not have a `__name__` attribute.', INVALID_PARAMETER_VALUE)
        name = eval_fn.__name__
    if '/' in name:
        raise MlflowException(f"Invalid metric name '{name}'. Metric names cannot include forward slashes ('/').", INVALID_PARAMETER_VALUE)
    if not name.isidentifier():
        _logger.warning(f"The metric name '{name}' provided is not a valid Python identifier, which will prevent its use as a base metric for derived metrics. Please use a valid identifier to enable creation of derived metrics that use the given metric.")
    if keyword.iskeyword(name):
        _logger.warning(f"The metric name '{name}' is a reserved Python keyword, which will prevent its use as a base metric for derived metrics. Please use a valid identifier to enable creation of derived metrics that use the given metric.")
    if name in ['predictions', 'targets', 'metrics']:
        _logger.warning(f"The metric name '{name}' is used as a special parameter in MLflow metrics, which will prevent its use as a base metric for derived metrics. Please use a different name to enable creation of derived metrics that use the given metric.")
    return EvaluationMetric(eval_fn, name, greater_is_better, long_name, version, metric_details)