import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def should_log_post_training_metrics(self):
    """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safe guard checking,
        See following note.

        Note: It includes checking `_SparkTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. CrossValidator/TrainValidationSplit) case:
          running CrossValidator.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
    return not _SparkTrainingSession.is_active() and self._log_post_training_metrics_enabled