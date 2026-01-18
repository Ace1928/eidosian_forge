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
def maybe_promote(dtype: np.dtype, fill_value=np.nan):
    """
    Find the minimal dtype that can hold both the given dtype and fill_value.

    Parameters
    ----------
    dtype : np.dtype
    fill_value : scalar, default np.nan

    Returns
    -------
    dtype
        Upcasted from dtype argument if necessary.
    fill_value
        Upcasted from fill_value argument if necessary.

    Raises
    ------
    ValueError
        If fill_value is a non-scalar and dtype is not object.
    """
    orig = fill_value
    orig_is_nat = False
    if checknull(fill_value):
        if fill_value is not NA:
            try:
                orig_is_nat = np.isnat(fill_value)
            except TypeError:
                pass
        fill_value = _canonical_nans.get(type(fill_value), fill_value)
    try:
        dtype, fill_value = _maybe_promote_cached(dtype, fill_value, type(fill_value))
    except TypeError:
        dtype, fill_value = _maybe_promote(dtype, fill_value)
    if dtype == _dtype_obj and orig is not None or (orig_is_nat and np.datetime_data(orig)[0] != 'ns'):
        fill_value = orig
    return (dtype, fill_value)