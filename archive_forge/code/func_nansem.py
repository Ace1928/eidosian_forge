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
def nansem(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, ddof: int=1, mask: npt.NDArray[np.bool_] | None=None) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
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
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     0.5773502691896258
    """
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != 'f':
        values = values.astype('f8')
    if not skipna and mask is not None and mask.any():
        return np.nan
    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    return np.sqrt(var) / np.sqrt(count)