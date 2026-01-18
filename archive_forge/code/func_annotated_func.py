import textwrap
import warnings
from functools import wraps
from typing import Dict
import importlib_metadata
from packaging.version import Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
imports declared from a common root path if multiple files are defined with import dependencies
def annotated_func(func):
    _, module_key = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[integration_name]
    min_ver, max_ver = get_module_min_and_max_supported_ranges(module_key)
    required_pkg_versions = f'``{min_ver}`` -  ``{max_ver}``'
    notice = f"The '{integration_name}' MLflow Models integration is known to be compatible with the following package version ranges: {required_pkg_versions}. MLflow Models integrations with {integration_name} may not succeed when used with package versions outside of this range."

    @wraps(func)
    def version_func(*args, **kwargs):
        installed_version = Version(importlib_metadata.version(module_key))
        if installed_version < Version(min_ver) or installed_version > Version(max_ver):
            warnings.warn(notice, category=FutureWarning, stacklevel=2)
        return func(*args, **kwargs)
    version_func.__doc__ = '    .. Note:: ' + notice + '\n' * 2 + func.__doc__ if func.__doc__ else notice
    return version_func