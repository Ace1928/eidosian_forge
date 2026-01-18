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
def maybe_cast_to_integer_array(arr: list | np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.

    Parameters
    ----------
    arr : np.ndarray or list
        The array to cast.
    dtype : np.dtype
        The integer dtype to cast the array to.

    Returns
    -------
    ndarray
        Array of integer or unsigned integer dtype.

    Raises
    ------
    OverflowError : the dtype is incompatible with the data
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> maybe_cast_to_integer_array([1, 2, 3.5], dtype=np.dtype("int64"))
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    """
    assert dtype.kind in 'iu'
    try:
        if not isinstance(arr, np.ndarray):
            with warnings.catch_warnings():
                if not np_version_gt2:
                    warnings.filterwarnings('ignore', 'NumPy will stop allowing conversion of out-of-bound Python int', DeprecationWarning)
                casted = np.array(arr, dtype=dtype, copy=False)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                casted = arr.astype(dtype, copy=False)
    except OverflowError as err:
        raise OverflowError(f'The elements provided in the data cannot all be casted to the dtype {dtype}') from err
    if isinstance(arr, np.ndarray) and arr.dtype == dtype:
        return casted
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', 'elementwise comparison failed', FutureWarning)
        if np.array_equal(arr, casted):
            return casted
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, str):
        if (casted.astype(str) == arr).all():
            return casted
        raise ValueError(f'string values cannot be losslessly cast to {dtype}')
    if dtype.kind == 'u' and (arr < 0).any():
        raise OverflowError('Trying to coerce negative values to unsigned integers')
    if arr.dtype.kind == 'f':
        if not np.isfinite(arr).all():
            raise IntCastingNaNError('Cannot convert non-finite values (NA or inf) to integer')
        raise ValueError('Trying to coerce float values to integers')
    if arr.dtype == object:
        raise ValueError('Trying to coerce float values to integers')
    if casted.dtype < arr.dtype:
        raise ValueError(f'Values are too large to be losslessly converted to {dtype}. To cast anyway, use pd.Series(values).astype({dtype})')
    if arr.dtype.kind in 'mM':
        raise TypeError(f'Constructing a Series or DataFrame from {arr.dtype} values and dtype={dtype} is not supported. Use values.view({dtype}) instead.')
    raise ValueError(f'values cannot be losslessly cast to {dtype}')