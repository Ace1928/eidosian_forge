from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def construct_2d_arraylike_from_scalar(value: Scalar, length: int, width: int, dtype: np.dtype, copy: bool) -> np.ndarray:
    shape = (length, width)
    if dtype.kind in 'mM':
        value = _maybe_box_and_unbox_datetimelike(value, dtype)
    elif dtype == _dtype_obj:
        if isinstance(value, (np.timedelta64, np.datetime64)):
            out = np.empty(shape, dtype=object)
            out.fill(value)
            return out
    try:
        arr = np.array(value, dtype=dtype, copy=copy)
    except (ValueError, TypeError) as err:
        raise TypeError(f'DataFrame constructor called with incompatible data and dtype: {err}') from err
    if arr.ndim != 0:
        raise ValueError('DataFrame constructor not properly called!')
    return np.full(shape, arr)