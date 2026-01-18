from __future__ import annotations
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool=False) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
    if values.dtype == dtype:
        if copy:
            return values.copy()
        return values
    if not isinstance(values, np.ndarray):
        values = values.astype(dtype, copy=copy)
    else:
        values = _astype_nansafe(values, dtype, copy=copy)
    if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
        values = np.array(values, dtype=object)
    return values