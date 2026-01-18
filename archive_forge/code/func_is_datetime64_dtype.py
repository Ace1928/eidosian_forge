from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def is_datetime64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_dtype
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype.kind == 'M'
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64))