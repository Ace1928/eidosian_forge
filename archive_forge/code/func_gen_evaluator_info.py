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
def gen_evaluator_info(self, evaluator):
    """
        Generate evaluator information, include evaluator class name and params.
        """
    class_name = _get_fully_qualified_class_name(evaluator)
    param_map = _truncate_dict(_get_param_map(evaluator), MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
    return {'evaluator_class': class_name, 'params': param_map}