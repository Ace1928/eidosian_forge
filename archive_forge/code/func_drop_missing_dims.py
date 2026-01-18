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
def drop_missing_dims(supplied_dims: Iterable[_Dim], dims: Iterable[_Dim], missing_dims: ErrorOptionsWithWarn) -> _DimsLike:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : Iterable of Hashable
    dims : Iterable of Hashable
    missing_dims : {"raise", "warn", "ignore"}
    """
    if missing_dims == 'raise':
        supplied_dims_set = {val for val in supplied_dims if val is not ...}
        if (invalid := (supplied_dims_set - set(dims))):
            raise ValueError(f'Dimensions {invalid} do not exist. Expected one or more of {dims}')
        return supplied_dims
    elif missing_dims == 'warn':
        if (invalid := (set(supplied_dims) - set(dims))):
            warnings.warn(f'Dimensions {invalid} do not exist. Expected one or more of {dims}')
        return [val for val in supplied_dims if val in dims or val is ...]
    elif missing_dims == 'ignore':
        return [val for val in supplied_dims if val in dims or val is ...]
    else:
        raise ValueError(f'Unrecognised option {missing_dims} for missing_dims argument')