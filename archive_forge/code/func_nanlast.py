from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def nanlast(values, axis, keepdims=False):
    if isinstance(axis, tuple):
        axis, = axis
    axis = normalize_axis_index(axis, values.ndim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(values)[rev], axis=axis)
    result = _select_along_axis(values, idx_last, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result