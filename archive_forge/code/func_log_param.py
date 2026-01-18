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
def log_param(self, run_id, key, value, synchronous=True):
    """Log a parameter (e.g. model hyperparameter) against the run ID. Value is converted to
        a string.

        Args:
            run_id: ID of the run to log the parameter against.
            key: Name of the parameter.
            value: Value of the parameter.
            synchronous: *Experimental* If True, blocks until the parameters are logged
                successfully. If False, logs the parameters asynchronously and
                returns a future representing the logging operation.

        Returns:
            When synchronous=True, returns parameter value.
            When synchronous=False, returns :py:class:`mlflow.RunOperations` that
            represents future for logging operation.

        """
    param = Param(key, str(value))
    try:
        if synchronous:
            self.store.log_param(run_id, param)
            return value
        else:
            return self.store.log_param_async(run_id, param)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE):
            msg = f'{e.message}{PARAM_VALIDATION_MSG}'
            raise MlflowException(msg, INVALID_PARAMETER_VALUE)
        else:
            raise e