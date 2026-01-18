from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like
from pandas.core.arrays import ExtensionArray
def setitem_datetimelike_compat(values: np.ndarray, num_set: int, other):
    """
    Parameters
    ----------
    values : np.ndarray
    num_set : int
        For putmask, this is mask.sum()
    other : Any
    """
    if values.dtype == object:
        dtype, _ = infer_dtype_from(other)
        if lib.is_np_dtype(dtype, 'mM'):
            if not is_list_like(other):
                other = [other] * num_set
            else:
                other = list(other)
    return other