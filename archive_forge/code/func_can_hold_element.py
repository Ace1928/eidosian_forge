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
def can_hold_element(arr: ArrayLike, element: Any) -> bool:
    """
    Can we do an inplace setitem with this element in an array with this dtype?

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
    element : Any

    Returns
    -------
    bool
    """
    dtype = arr.dtype
    if not isinstance(dtype, np.dtype) or dtype.kind in 'mM':
        if isinstance(dtype, (PeriodDtype, IntervalDtype, DatetimeTZDtype, np.dtype)):
            arr = cast('PeriodArray | DatetimeArray | TimedeltaArray | IntervalArray', arr)
            try:
                arr._validate_setitem_value(element)
                return True
            except (ValueError, TypeError):
                return False
        return True
    try:
        np_can_hold_element(dtype, element)
        return True
    except (TypeError, LossySetitemError):
        return False