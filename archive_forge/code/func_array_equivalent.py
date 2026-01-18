from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def array_equivalent(left, right, strict_nan: bool=False, dtype_equal: bool=False) -> bool:
    """
    True if two arrays, left and right, have equal non-NaN elements, and NaNs
    in corresponding locations.  False otherwise. It is assumed that left and
    right are NumPy arrays of the same dtype. The behavior of this function
    (particularly with respect to NaNs) is not defined if the dtypes are
    different.

    Parameters
    ----------
    left, right : ndarrays
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    dtype_equal : bool, default False
        Whether `left` and `right` are known to have the same dtype
        according to `is_dtype_equal`. Some methods like `BlockManager.equals`.
        require that the dtypes match. Setting this to ``True`` can improve
        performance, but will give different results for arrays that are
        equal but different dtypes.

    Returns
    -------
    b : bool
        Returns True if the arrays are equivalent.

    Examples
    --------
    >>> array_equivalent(
    ...     np.array([1, 2, np.nan]),
    ...     np.array([1, 2, np.nan]))
    True
    >>> array_equivalent(
    ...     np.array([1, np.nan, 2]),
    ...     np.array([1, 2, np.nan]))
    False
    """
    left, right = (np.asarray(left), np.asarray(right))
    if left.shape != right.shape:
        return False
    if dtype_equal:
        if left.dtype.kind in 'fc':
            return _array_equivalent_float(left, right)
        elif left.dtype.kind in 'mM':
            return _array_equivalent_datetimelike(left, right)
        elif is_string_or_object_np_dtype(left.dtype):
            return _array_equivalent_object(left, right, strict_nan)
        else:
            return np.array_equal(left, right)
    if left.dtype.kind in 'OSU' or right.dtype.kind in 'OSU':
        return _array_equivalent_object(left, right, strict_nan)
    if left.dtype.kind in 'fc':
        if not (left.size and right.size):
            return True
        return ((left == right) | isna(left) & isna(right)).all()
    elif left.dtype.kind in 'mM' or right.dtype.kind in 'mM':
        if left.dtype != right.dtype:
            return False
        left = left.view('i8')
        right = right.view('i8')
    if (left.dtype.type is np.void or right.dtype.type is np.void) and left.dtype != right.dtype:
        return False
    return np.array_equal(left, right)