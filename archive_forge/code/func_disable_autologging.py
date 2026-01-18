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
@contextlib.contextmanager
def disable_autologging():
    """
    Context manager that temporarily disables autologging globally for all integrations upon
    entry and restores the previous autologging configuration upon exit.
    """
    global _AUTOLOGGING_GLOBALLY_DISABLED
    _AUTOLOGGING_GLOBALLY_DISABLED = True
    try:
        yield None
    finally:
        _AUTOLOGGING_GLOBALLY_DISABLED = False