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
def patched_IffHasAttrDescriptor__get__(self, obj, type=None):
    """
                For sklearn version <= 0.24.2, `_IffHasAttrDescriptor.__get__` method does not
                support unbound method call.
                See https://github.com/scikit-learn/scikit-learn/issues/20614
                This patched function is for hot patch.
                """
    if obj is not None:
        for delegate_name in self.delegate_names:
            try:
                delegate = sklearn.utils.metaestimators.attrgetter(delegate_name)(obj)
            except AttributeError:
                continue
            else:
                getattr(delegate, self.attribute_name)
                break
        else:
            sklearn.utils.metaestimators.attrgetter(self.delegate_names[-1])(obj)

        def out(*args, **kwargs):
            return self.fn(obj, *args, **kwargs)
    else:

        def out(*args, **kwargs):
            return self.fn(*args, **kwargs)
    functools.update_wrapper(out, self.fn)
    return out