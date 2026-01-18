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
def func_interpolate_na(interpolator, y, x, **kwargs):
    """helper function to apply interpolation along 1 dimension"""
    out = y.copy()
    nans = pd.isnull(y)
    nonans = ~nans
    n_nans = nans.sum()
    if n_nans == 0 or n_nans >= len(y) - 1:
        return y
    f = interpolator(x[nonans], y[nonans], **kwargs)
    out[nans] = f(x[nans])
    return out