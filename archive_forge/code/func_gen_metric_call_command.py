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
@staticmethod
def gen_metric_call_command(self_obj, metric_fn, *call_pos_args, **call_kwargs):
    """
        Generate metric function call command string like `metric_fn(arg1, arg2, ...)`
        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.

        Args:
            self_obj: If the metric_fn is a method of an instance (e.g. `model.score`),
            the `self_obj` represent the instance.
            metric_fn: metric function.
            call_pos_args: the positional arguments of the metric function call. If `metric_fn`
            is instance method, then the `call_pos_args` should exclude the first `self` argument.
            call_kwargs: the keyword arguments of the metric function call.
        """
    arg_list = []

    def arg_to_str(arg):
        if arg is None or np.isscalar(arg):
            if isinstance(arg, str) and len(arg) > 32:
                return repr(arg[:32] + '...')
            return repr(arg)
        else:
            return _inspect_original_var_name(arg, fallback_name=f'<{arg.__class__.__name__}>')
    param_sig = inspect.signature(metric_fn).parameters
    arg_names = list(param_sig.keys())
    if self_obj is not None:
        arg_names.pop(0)
    if self_obj is not None:
        call_fn_name = f'{self_obj.__class__.__name__}.{metric_fn.__name__}'
    else:
        call_fn_name = metric_fn.__name__
    for arg_name, arg in zip(arg_names, call_pos_args):
        arg_list.append(f'{arg_name}={arg_to_str(arg)}')
    for arg_name, arg in call_kwargs.items():
        arg_list.append(f'{arg_name}={arg_to_str(arg)}')
    arg_list_str = ', '.join(arg_list)
    return f'{call_fn_name}({arg_list_str})'