from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def get_squeeze_dims(xarray_obj, dim: Hashable | Iterable[Hashable] | None=None, axis: int | Iterable[int] | None=None) -> list[Hashable]:
    """Get a list of dimensions to squeeze out."""
    if dim is not None and axis is not None:
        raise ValueError('cannot use both parameters `axis` and `dim`')
    if dim is None and axis is None:
        return [d for d, s in xarray_obj.sizes.items() if s == 1]
    if isinstance(dim, Iterable) and (not isinstance(dim, str)):
        dim = list(dim)
    elif dim is not None:
        dim = [dim]
    else:
        assert axis is not None
        if isinstance(axis, int):
            axis = [axis]
        axis = list(axis)
        if any((not isinstance(a, int) for a in axis)):
            raise TypeError('parameter `axis` must be int or iterable of int.')
        alldims = list(xarray_obj.sizes.keys())
        dim = [alldims[a] for a in axis]
    if any((xarray_obj.sizes[k] > 1 for k in dim)):
        raise ValueError('cannot select a dimension to squeeze out which has length greater than one')
    return dim