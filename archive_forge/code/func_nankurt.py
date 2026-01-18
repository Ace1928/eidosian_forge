from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
@disallow('M8', 'm8')
@maybe_operate_rowwise
def nankurt(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s.values)
    -1.2892561983471076
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != 'f':
        values = values.astype('f8')
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)
    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted ** 2
    adjusted4 = adjusted2 ** 2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2 ** 2
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)
    if not isinstance(denominator, np.ndarray):
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = numerator / denominator - adj
    dtype = values.dtype
    if dtype.kind == 'f':
        result = result.astype(dtype, copy=False)
    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan
    return result