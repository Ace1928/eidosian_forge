import functools
import importlib
import os
import warnings
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar
import pkg_resources
from packaging.requirements import Requirement
from packaging.version import Version
from typing_extensions import ParamSpec
@lru_cache
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False

    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False