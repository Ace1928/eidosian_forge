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
@staticmethod
def get_current_build_dependencies():
    build_dependencies = []
    for package in _PythonEnv.BUILD_PACKAGES:
        version = _PythonEnv._get_package_version(package)
        dep = package + '==' + version if version else package
        build_dependencies.append(dep)
    return build_dependencies