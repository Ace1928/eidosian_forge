import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
def get_underlying_model_flavor(model):
    """
    Find the underlying models flavor.

    Args:
        model: underlying model of the explainer.
    """
    if hasattr(model, 'inner_model'):
        unwrapped_model = model.inner_model
        if isinstance(unwrapped_model, types.MethodType):
            model_object = unwrapped_model.__self__
            try:
                import sklearn
                if issubclass(type(model_object), sklearn.base.BaseEstimator):
                    return mlflow.sklearn.FLAVOR_NAME
            except ImportError:
                pass
        try:
            import torch
            if issubclass(type(unwrapped_model), torch.nn.Module):
                return mlflow.pytorch.FLAVOR_NAME
        except ImportError:
            pass
    return _UNKNOWN_MODEL_FLAVOR