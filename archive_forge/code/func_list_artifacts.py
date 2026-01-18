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
def list_artifacts(self, run_id, path=None):
    """List the artifacts for a run.

        Args:
            run_id: The run to list artifacts from.
            path: The run's relative artifact path to list from. By default it is set to None
                or the root artifact path.

        Returns:
            List of :py:class:`mlflow.entities.FileInfo`

        """
    return self._get_artifact_repo(run_id).list_artifacts(path)