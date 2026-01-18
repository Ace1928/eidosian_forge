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
def infer_pip_requirements_with_timeout(model_uri, flavor, fallback):
    timeout = MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT.get()
    if timeout and is_windows():
        timeout = None
        _logger.warning('On Windows, timeout is not supported for model requirement inference. Therefore, the operation is not bound by a timeout and may hang indefinitely. If it hangs, please consider specifying the signature manually.')
    try:
        if timeout:
            with run_with_timeout(timeout):
                return infer_pip_requirements(model_uri, flavor, fallback)
        else:
            return infer_pip_requirements(model_uri, flavor, fallback)
    except Exception as e:
        if fallback is not None:
            if isinstance(e, MlflowTimeoutError):
                msg = f'Attempted to infer pip requirements for the saved model or pipeline but the operation timed out in {timeout} seconds. Fall back to return {fallback}. You can specify a different timeout by setting the environment variable {MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}.'
            else:
                msg = _INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE.format(model_uri=model_uri, flavor=flavor, fallback=fallback)
            _logger.warning(msg)
            _logger.debug('', exc_info=True)
            return fallback
        raise