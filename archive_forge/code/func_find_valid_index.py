from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def find_valid_index(how: str, is_valid: npt.NDArray[np.bool_]) -> int | None:
    """
    Retrieves the positional index of the first valid value.

    Parameters
    ----------
    how : {'first', 'last'}
        Use this parameter to change between the first or last valid index.
    is_valid: np.ndarray
        Mask to find na_values.

    Returns
    -------
    int or None
    """
    assert how in ['first', 'last']
    if len(is_valid) == 0:
        return None
    if is_valid.ndim == 2:
        is_valid = is_valid.any(axis=1)
    if how == 'first':
        idxpos = is_valid[:].argmax()
    elif how == 'last':
        idxpos = len(is_valid) - 1 - is_valid[::-1].argmax()
    chk_notna = is_valid[idxpos]
    if not chk_notna:
        return None
    return idxpos