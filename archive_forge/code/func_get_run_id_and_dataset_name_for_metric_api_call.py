import functools
import inspect
import logging
import os
import pickle
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _inspect_original_var_name, gorilla
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def get_run_id_and_dataset_name_for_metric_api_call(self, call_pos_args, call_kwargs):
    """
        Given a metric api call (include the called metric function, and call arguments)
        Register the call information (arguments dict) into the `metric_api_call_arg_dict_list_map`
        and return a tuple of (run_id, eval_dataset_name)
        """
    call_arg_list = list(call_pos_args) + list(call_kwargs.values())
    dataset_id_list = self._pred_result_id_to_dataset_name_and_run_id.keys()
    for arg in call_arg_list:
        if arg is not None and (not np.isscalar(arg)) and (id(arg) in dataset_id_list):
            dataset_name, run_id = self._pred_result_id_to_dataset_name_and_run_id[id(arg)]
            break
    else:
        return (None, None)
    return (run_id, dataset_name)