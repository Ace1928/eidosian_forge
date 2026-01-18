from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def as_indexable(array):
    """
    This function always returns a ExplicitlyIndexed subclass,
    so that the vectorized indexing is always possible with the returned
    object.
    """
    if isinstance(array, ExplicitlyIndexed):
        return array
    if isinstance(array, np.ndarray):
        return NumpyIndexingAdapter(array)
    if isinstance(array, pd.Index):
        return PandasIndexingAdapter(array)
    if is_duck_dask_array(array):
        return DaskIndexingAdapter(array)
    if hasattr(array, '__array_function__'):
        return NdArrayLikeIndexingAdapter(array)
    if hasattr(array, '__array_namespace__'):
        return ArrayApiIndexingAdapter(array)
    raise TypeError(f'Invalid array type: {type(array)}')