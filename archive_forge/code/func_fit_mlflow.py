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
def fit_mlflow(original, self, *args, **kwargs):
    params = get_method_call_arg_value(1, 'params', None, args, kwargs)
    if _get_fully_qualified_class_name(self).startswith('pyspark.ml.feature.'):
        return original(self, *args, **kwargs)
    elif isinstance(params, (list, tuple)):
        _logger.warning(_get_warning_msg_for_fit_call_with_a_list_of_params(self))
        return original(self, *args, **kwargs)
    else:
        from pyspark.storagelevel import StorageLevel
        estimator = self.copy(params) if params is not None else self
        input_training_df = args[0].persist(StorageLevel.MEMORY_AND_DISK)
        _log_pretraining_metadata(estimator, params, input_training_df)
        spark_model = original(self, *args, **kwargs)
        _log_posttraining_metadata(estimator, spark_model, params, input_training_df)
        input_training_df.unpersist()
        return spark_model