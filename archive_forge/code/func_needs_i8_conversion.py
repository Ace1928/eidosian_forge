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
def needs_i8_conversion(dtype: DtypeObj | None) -> bool:
    """
    Check whether the dtype should be converted to int64.

    Dtype "needs" such a conversion if the dtype is of a datetime-like dtype

    Parameters
    ----------
    dtype : np.dtype, ExtensionDtype, or None

    Returns
    -------
    boolean
        Whether or not the dtype should be converted to int64.

    Examples
    --------
    >>> needs_i8_conversion(str)
    False
    >>> needs_i8_conversion(np.int64)
    False
    >>> needs_i8_conversion(np.datetime64)
    False
    >>> needs_i8_conversion(np.dtype(np.datetime64))
    True
    >>> needs_i8_conversion(np.array(['a', 'b']))
    False
    >>> needs_i8_conversion(pd.Series([1, 2]))
    False
    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern").dtype)
    True
    """
    if isinstance(dtype, np.dtype):
        return dtype.kind in 'mM'
    return isinstance(dtype, (PeriodDtype, DatetimeTZDtype))