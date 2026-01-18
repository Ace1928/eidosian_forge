import textwrap
import warnings
from functools import wraps
from typing import Dict
import importlib_metadata
from packaging.version import Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
imports declared from a common root path if multiple files are defined with import dependencies
def get_module_min_and_max_supported_ranges(module_name):
    """
    Extracts the minimum and maximum supported package versions from the provided module name.
    The version information is provided via the yaml-to-python-script generation script in
    dev/update_ml_package_versions.py which writes a python file to the importable namespace of
    mlflow.ml_package_versions

    Args:
        module_name: The string name of the module as it is registered in ml_package_versions.py

    Returns:
        tuple of minimum supported version, maximum supported version as strings.
    """
    versions = _ML_PACKAGE_VERSIONS[module_name]['models']
    min_version = versions['minimum']
    max_version = versions['maximum']
    return (min_version, max_version)