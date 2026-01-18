import importlib
import re
from packaging.version import InvalidVersion, Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def is_flavor_supported_for_associated_package_versions(flavor_name):
    """
    Returns:
        True if the specified flavor is supported for the currently-installed versions of its
        associated packages.
    """
    module_name, module_key = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[flavor_name]
    actual_version = importlib.import_module(module_name).__version__
    if module_name == 'pyspark' and is_in_databricks_runtime():
        actual_version = _strip_dev_version_suffix(actual_version)
    if _violates_pep_440(actual_version) or _is_pre_or_dev_release(actual_version):
        return False
    min_version, max_version, _ = get_min_max_version_and_pip_release(module_key)
    if module_name == 'pyspark' and is_in_databricks_runtime():
        if Version(max_version) < Version('3.3.0'):
            max_version = '3.3.0'
        return _check_spark_version_in_range(actual_version, min_version, max_version)
    else:
        return _check_version_in_range(actual_version, min_version, max_version)