import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import List
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import PYTHON_VERSION, insecure_hash
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.utils.requirements_utils import (
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
from mlflow.version import VERSION
def infer_pip_requirements(model_uri, flavor, fallback=None):
    """Infers the pip requirements of the specified model by creating a subprocess and loading
    the model in it to determine which packages are imported.

    Args:
        model_uri: The URI of the model.
        flavor: The flavor name of the model.
        fallback: If provided, an unexpected error during the inference procedure is swallowed
            and the value of ``fallback`` is returned. Otherwise, the error is raised.

    Returns:
        A list of inferred pip requirements (e.g. ``["scikit-learn==0.24.2", ...]``).

    """
    try:
        return _infer_requirements(model_uri, flavor)
    except Exception:
        if fallback is not None:
            _logger.warning(msg=_INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE.format(model_uri=model_uri, flavor=flavor, fallback=fallback))
            _logger.debug('', exc_info=True)
            return fallback
        raise