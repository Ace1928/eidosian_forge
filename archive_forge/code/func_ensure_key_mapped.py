from __future__ import annotations
from collections import defaultdict
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
def ensure_key_mapped(values: ArrayLike | Index | Series, key: Callable | None, levels=None) -> ArrayLike | Index | Series:
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
    from pandas.core.indexes.api import Index
    if not key:
        return values
    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)
    result = key(values.copy())
    if len(result) != len(values):
        raise ValueError('User-provided `key` function must not change the shape of the array.')
    try:
        if isinstance(values, Index):
            result = Index(result)
        else:
            type_of_values = type(values)
            result = type_of_values(result)
    except TypeError:
        raise TypeError(f'User-provided `key` function returned an invalid type {type(result)}             which could not be converted to {type(values)}.')
    return result