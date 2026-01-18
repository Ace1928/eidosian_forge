from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import (
def coerce_to_array(values, mask=None, copy: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    if isinstance(values, BooleanArray):
        if mask is not None:
            raise ValueError('cannot pass mask for BooleanArray input')
        values, mask = (values._data, values._mask)
        if copy:
            values = values.copy()
            mask = mask.copy()
        return (values, mask)
    mask_values = None
    if isinstance(values, np.ndarray) and values.dtype == np.bool_:
        if copy:
            values = values.copy()
    elif isinstance(values, np.ndarray) and values.dtype.kind in 'iufcb':
        mask_values = isna(values)
        values_bool = np.zeros(len(values), dtype=bool)
        values_bool[~mask_values] = values[~mask_values].astype(bool)
        if not np.all(values_bool[~mask_values].astype(values.dtype) == values[~mask_values]):
            raise TypeError('Need to pass bool-like values')
        values = values_bool
    else:
        values_object = np.asarray(values, dtype=object)
        inferred_dtype = lib.infer_dtype(values_object, skipna=True)
        integer_like = ('floating', 'integer', 'mixed-integer-float')
        if inferred_dtype not in ('boolean', 'empty') + integer_like:
            raise TypeError('Need to pass bool-like values')
        mask_values = cast('npt.NDArray[np.bool_]', isna(values_object))
        values = np.zeros(len(values), dtype=bool)
        values[~mask_values] = values_object[~mask_values].astype(bool)
        if inferred_dtype in integer_like and (not np.all(values[~mask_values].astype(float) == values_object[~mask_values].astype(float))):
            raise TypeError('Need to pass bool-like values')
    if mask is None and mask_values is None:
        mask = np.zeros(values.shape, dtype=bool)
    elif mask is None:
        mask = mask_values
    elif isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
        if mask_values is not None:
            mask = mask | mask_values
        elif copy:
            mask = mask.copy()
    else:
        mask = np.array(mask, dtype=bool)
        if mask_values is not None:
            mask = mask | mask_values
    if values.shape != mask.shape:
        raise ValueError('values.shape and mask.shape must match')
    return (values, mask)