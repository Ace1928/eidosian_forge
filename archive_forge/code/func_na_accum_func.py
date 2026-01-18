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
def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    mask_a, mask_b = {np.cumprod: (1.0, np.nan), np.maximum.accumulate: (-np.inf, np.nan), np.cumsum: (0.0, np.nan), np.minimum.accumulate: (np.inf, np.nan)}[accum_func]
    assert values.dtype.kind not in 'mM'
    if skipna and (not issubclass(values.dtype.type, (np.integer, np.bool_))):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)
    return result