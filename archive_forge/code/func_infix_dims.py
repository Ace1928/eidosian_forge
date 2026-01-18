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
def infix_dims(dims_supplied: Iterable[_Dim], dims_all: Iterable[_Dim], missing_dims: ErrorOptionsWithWarn='raise') -> Iterator[_Dim]:
    """
    Resolves a supplied list containing an ellipsis representing other items, to
    a generator with the 'realized' list of all items
    """
    if ... in dims_supplied:
        dims_all_list = list(dims_all)
        if len(set(dims_all)) != len(dims_all_list):
            raise ValueError('Cannot use ellipsis with repeated dims')
        if list(dims_supplied).count(...) > 1:
            raise ValueError('More than one ellipsis supplied')
        other_dims = [d for d in dims_all if d not in dims_supplied]
        existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
        for d in existing_dims:
            if d is ...:
                yield from other_dims
            else:
                yield d
    else:
        existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
        if set(existing_dims) ^ set(dims_all):
            raise ValueError(f'{dims_supplied} must be a permuted list of {dims_all}, unless `...` is included')
        yield from existing_dims