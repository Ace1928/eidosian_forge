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
def get_method_call_arg_value(arg_index, arg_name, default_value, call_pos_args, call_kwargs):
    """Get argument value for a method call.

    Args:
        arg_index: The argument index in the function signature. Start from 0.
        arg_name: The argument name in the function signature.
        default_value: Default argument value.
        call_pos_args: The positional argument values in the method call.
        call_kwargs: The keyword argument values in the method call.
    """
    if arg_name in call_kwargs:
        return call_kwargs[arg_name]
    elif arg_index < len(call_pos_args):
        return call_pos_args[arg_index]
    else:
        return default_value