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
def either_dict_or_kwargs(pos_kwargs: Mapping[Any, T] | None, kw_kwargs: Mapping[str, T], func_name: str) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        return cast(Mapping[Hashable, T], kw_kwargs)
    if not is_dict_like(pos_kwargs):
        raise ValueError(f'the first argument to .{func_name} must be a dictionary')
    if kw_kwargs:
        raise ValueError(f'cannot specify both keyword and positional arguments to .{func_name}')
    return pos_kwargs