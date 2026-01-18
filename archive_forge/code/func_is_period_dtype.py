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
def is_period_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Period dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.Period) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Period dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_period_dtype
    >>> is_period_dtype(object)
    False
    >>> is_period_dtype(pd.PeriodDtype(freq="D"))
    True
    >>> is_period_dtype([1, 2, 3])
    False
    >>> is_period_dtype(pd.Period("2017-01-01"))
    False
    >>> is_period_dtype(pd.PeriodIndex([], freq="Y"))
    True
    """
    warnings.warn('is_period_dtype is deprecated and will be removed in a future version. Use `isinstance(dtype, pd.PeriodDtype)` instead', DeprecationWarning, stacklevel=2)
    if isinstance(arr_or_dtype, ExtensionDtype):
        return arr_or_dtype.type is Period
    if arr_or_dtype is None:
        return False
    return PeriodDtype.is_dtype(arr_or_dtype)