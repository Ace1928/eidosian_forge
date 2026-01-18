import time
from contextlib import contextmanager
import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.metrics_queue import (
def patched_add_summary(original, self, *args, **kwargs):
    flush_metrics_queue()
    return original(self, *args, **kwargs)