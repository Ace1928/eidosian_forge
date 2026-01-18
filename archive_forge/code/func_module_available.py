from __future__ import annotations
import importlib
import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast
import numpy as np
from packaging.version import Version
from xarray.namedarray._typing import ErrorOptionsWithWarn, _DimsLike
@lru_cache
def module_available(module: str, minversion: str | None=None) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    if importlib.util.find_spec(module) is None:
        return False
    if minversion is not None:
        version = importlib.metadata.version(module)
        return Version(version) >= Version(minversion)
    return True