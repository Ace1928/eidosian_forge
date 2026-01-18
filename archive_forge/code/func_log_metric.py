import os
from collections import OrderedDict
from itertools import zip_longest
from typing import List, Optional
from mlflow.entities import ExperimentTag, Metric, Param, RunStatus, RunTag, ViewType
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking import GET_METRIC_HISTORY_MAX_RESULTS, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._tracking_service import utils
from mlflow.tracking.metric_value_conversion_utils import convert_metric_value_to_float_if_possible
from mlflow.utils import chunk_list
from mlflow.utils.async_logging.run_operations import RunOperations, get_combined_run_operations
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri
from mlflow.utils.validation import (
def log_metric(self, run_id, key, value, timestamp=None, step=None, synchronous=True) -> Optional[RunOperations]:
    """Log a metric against the run ID.

        Args:
            run_id: The run id to which the metric should be logged.
            key: Metric name. This string may only contain alphanumerics, underscores (_),
                dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores will
                support keys up to length 250, but some may support larger keys.
            value: Metric value or single-item ndarray / tensor. Note that some special values such
                as +/- Infinity may be replaced by other values depending on the store. For example,
                the SQLAlchemy store replaces +/- Inf with max / min float values. All backend
                stores will support values up to length 5000, but some may support larger values.
            timestamp: Time when this metric was calculated. Defaults to the current system time.
            step: Training step (iteration) at which was the metric calculated. Defaults to 0.
            synchronous: *Experimental* If True, blocks until the metric is logged successfully. If
                False, logs the metric asynchronously and returns a future representing the logging
                operation.

        Returns:
            When synchronous=True, returns None. When synchronous=False, returns
            :py:class:`mlflow.RunOperations` that represents future for logging operation.

        """
    timestamp = timestamp if timestamp is not None else get_current_time_millis()
    step = step if step is not None else 0
    metric_value = convert_metric_value_to_float_if_possible(value)
    metric = Metric(key, metric_value, timestamp, step)
    if synchronous:
        self.store.log_metric(run_id, metric)
    else:
        return self.store.log_metric_async(run_id, metric)