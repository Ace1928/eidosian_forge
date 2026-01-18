from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def convert_datetime_array(array: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]:
    """ Convert NumPy datetime arrays to arrays to milliseconds since epoch.

    Args:
        array : (obj)
            A NumPy array of datetime to convert

            If the value passed in is not a NumPy array, it will be returned as-is.

    Returns:
        array

    """

    def convert(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.where(np.isnat(array), np.nan, array.astype('int64') / 1000.0)
    if array.dtype.kind == 'M':
        return convert(array.astype('datetime64[us]'))
    elif array.dtype.kind == 'm':
        return convert(array.astype('timedelta64[us]'))
    elif array.dtype.kind == 'O' and len(array) > 0 and isinstance(array[0], dt.date):
        try:
            return convert(array.astype('datetime64[us]'))
        except Exception:
            pass
    return array