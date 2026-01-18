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
def is_dask_collection(x: object) -> TypeGuard[DaskCollection]:
    if module_available('dask'):
        from dask.base import is_dask_collection
        return is_dask_collection(x)
    return False