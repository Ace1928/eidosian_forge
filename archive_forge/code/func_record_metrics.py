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
def record_metrics(self, metrics, step=None):
    """
        Submit a set of metrics to be logged. The metrics may not be immediately logged, as this
        class will batch them in order to not increase execution time too much by logging
        frequently.

        Args:
            metrics: Dictionary containing key, value pairs of metrics to be logged.
            step: The training step that the metrics correspond to.
        """
    current_timestamp = time.time()
    if self.previous_training_timestamp is None:
        self.previous_training_timestamp = current_timestamp
    training_time = current_timestamp - self.previous_training_timestamp
    self.total_training_time += training_time
    if step is None:
        step = 0
    for key, value in metrics.items():
        self.data.append(Metric(key, value, int(current_timestamp * 1000), step))
    if self._should_flush():
        self.flush()
    self.previous_training_timestamp = current_timestamp