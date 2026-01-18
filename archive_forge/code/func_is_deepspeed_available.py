import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def is_deepspeed_available():
    package_exists = importlib.util.find_spec('deepspeed') is not None
    if package_exists:
        try:
            _ = importlib_metadata.metadata('deepspeed')
            return True
        except importlib_metadata.PackageNotFoundError:
            return False