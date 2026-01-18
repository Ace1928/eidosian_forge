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
def mask_missing(arr: ArrayLike, values_to_mask) -> npt.NDArray[np.bool_]:
    """
    Return a masking array of same size/shape as arr
    with entries equaling any member of values_to_mask set to True

    Parameters
    ----------
    arr : ArrayLike
    values_to_mask: list, tuple, or scalar

    Returns
    -------
    np.ndarray[bool]
    """
    dtype, values_to_mask = infer_dtype_from(values_to_mask)
    if isinstance(dtype, np.dtype):
        values_to_mask = np.array(values_to_mask, dtype=dtype)
    else:
        cls = dtype.construct_array_type()
        if not lib.is_list_like(values_to_mask):
            values_to_mask = [values_to_mask]
        values_to_mask = cls._from_sequence(values_to_mask, dtype=dtype, copy=False)
    potential_na = False
    if is_object_dtype(arr.dtype):
        potential_na = True
        arr_mask = ~isna(arr)
    na_mask = isna(values_to_mask)
    nonna = values_to_mask[~na_mask]
    mask = np.zeros(arr.shape, dtype=bool)
    if is_numeric_dtype(arr.dtype) and (not is_bool_dtype(arr.dtype)) and is_bool_dtype(nonna.dtype):
        pass
    elif is_bool_dtype(arr.dtype) and is_numeric_dtype(nonna.dtype) and (not is_bool_dtype(nonna.dtype)):
        pass
    else:
        for x in nonna:
            if is_numeric_v_string_like(arr, x):
                pass
            else:
                if potential_na:
                    new_mask = np.zeros(arr.shape, dtype=np.bool_)
                    new_mask[arr_mask] = arr[arr_mask] == x
                else:
                    new_mask = arr == x
                    if not isinstance(new_mask, np.ndarray):
                        new_mask = new_mask.to_numpy(dtype=bool, na_value=False)
                mask |= new_mask
    if na_mask.any():
        mask |= isna(arr)
    return mask