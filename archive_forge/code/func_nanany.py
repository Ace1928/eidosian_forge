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
def nanany(values: np.ndarray, *, axis: AxisInt | None=None, skipna: bool=True, mask: npt.NDArray[np.bool_] | None=None) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    False
    """
    if values.dtype.kind in 'iub' and mask is None:
        return values.any(axis)
    if values.dtype.kind == 'M':
        warnings.warn("'any' with datetime64 dtypes is deprecated and will raise in a future version. Use (obj != pd.Timestamp(0)).any() instead.", FutureWarning, stacklevel=find_stack_level())
    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return values.any(axis)