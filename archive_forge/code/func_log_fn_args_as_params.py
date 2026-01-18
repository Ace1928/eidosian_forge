import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def log_fn_args_as_params(fn, args, kwargs, unlogged=None):
    """Log arguments explicitly passed to a function as MLflow Run parameters to the current active
    MLflow Run.

    Args:
        fn: function whose parameters are to be logged
        args: arguments explicitly passed into fn. If `fn` is defined on a class,
            `self` should not be part of `args`; the caller is responsible for
            filtering out `self` before calling this function.
        kwargs: kwargs explicitly passed into fn
        unlogged: parameters not to be logged

    Returns:
        None

    """
    params_to_log = get_mlflow_run_params_for_fn_args(fn, args, kwargs, unlogged)
    mlflow.log_params(params_to_log)