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
def get_instance_method_first_arg_value(method, call_pos_args, call_kwargs):
    """Get instance method first argument value (exclude the `self` argument).

    Args:
        method: A `cls.method` object which includes the `self` argument.
        call_pos_args: positional arguments excluding the first `self` argument.
        call_kwargs: keywords arguments.
    """
    if len(call_pos_args) >= 1:
        return call_pos_args[0]
    else:
        param_sig = inspect.signature(method).parameters
        first_arg_name = list(param_sig.keys())[1]
        assert param_sig[first_arg_name].kind not in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]
        return call_kwargs.get(first_arg_name)