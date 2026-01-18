from __future__ import annotations
import decimal
import operator
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
from pandas.core.indexers import validate_indices
def factorize_array(values: np.ndarray, use_na_sentinel: bool=True, size_hint: int | None=None, na_value: object=None, mask: npt.NDArray[np.bool_] | None=None) -> tuple[npt.NDArray[np.intp], np.ndarray]:
    """
    Factorize a numpy array to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    size_hint : int, optional
        Passed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).
    mask : ndarray[bool], optional
        If not None, the mask is used as indicator for missing values
        (True = missing, False = valid) instead of `na_value` or
        condition "val != val".

    Returns
    -------
    codes : ndarray[np.intp]
    uniques : ndarray
    """
    original = values
    if values.dtype.kind in 'mM':
        na_value = iNaT
    hash_klass, values = _get_hashtable_algo(values)
    table = hash_klass(size_hint or len(values))
    uniques, codes = table.factorize(values, na_sentinel=-1, na_value=na_value, mask=mask, ignore_na=use_na_sentinel)
    uniques = _reconstruct_data(uniques, original.dtype, original)
    codes = ensure_platform_int(codes)
    return (codes, uniques)