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
def fit_mlflow_xgboost_and_lightgbm(original, self, *args, **kwargs):
    """
        Autologging function for XGBoost and LightGBM scikit-learn models
        """
    input_example_exc = None
    try:
        input_example = deepcopy(_get_X_y_and_sample_weight(self.fit, args, kwargs)[0][:INPUT_EXAMPLE_SAMPLE_ROWS])
    except Exception as e:
        input_example_exc = e

    def get_input_example():
        if input_example_exc is not None:
            raise input_example_exc
        else:
            return input_example
    fit_output = original(self, *args, **kwargs)
    if log_models:
        input_example, signature = resolve_input_example_and_signature(get_input_example, lambda input_example: infer_signature(input_example, self.predict(deepcopy(input_example))), log_input_examples, log_model_signatures, _logger)
        log_model_func = mlflow.xgboost.log_model if flavor_name == mlflow.xgboost.FLAVOR_NAME else mlflow.lightgbm.log_model
        registered_model_name = get_autologging_config(flavor_name, 'registered_model_name', None)
        if flavor_name == mlflow.xgboost.FLAVOR_NAME:
            model_format = get_autologging_config(flavor_name, 'model_format', 'xgb')
            log_model_func(self, artifact_path='model', signature=signature, input_example=input_example, registered_model_name=registered_model_name, model_format=model_format)
        else:
            log_model_func(self, artifact_path='model', signature=signature, input_example=input_example, registered_model_name=registered_model_name)
    return fit_output