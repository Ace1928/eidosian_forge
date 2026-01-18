from __future__ import annotations
import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args
import numpy as np
import pandas as pd
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def decompose_interp(indexes_coords):
    """Decompose the interpolation into a succession of independent interpolation keeping the order"""
    dest_dims = [dest[1].dims if dest[1].ndim > 0 else [dim] for dim, dest in indexes_coords.items()]
    partial_dest_dims = []
    partial_indexes_coords = {}
    for i, index_coords in enumerate(indexes_coords.items()):
        partial_indexes_coords.update([index_coords])
        if i == len(dest_dims) - 1:
            break
        partial_dest_dims += [dest_dims[i]]
        other_dims = dest_dims[i + 1:]
        s_partial_dest_dims = {dim for dims in partial_dest_dims for dim in dims}
        s_other_dims = {dim for dims in other_dims for dim in dims}
        if not s_partial_dest_dims.intersection(s_other_dims):
            yield partial_indexes_coords
            partial_dest_dims = []
            partial_indexes_coords = {}
    yield partial_indexes_coords