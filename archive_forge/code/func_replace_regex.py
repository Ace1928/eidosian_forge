from __future__ import annotations
import operator
import re
from re import Pattern
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
def replace_regex(values: ArrayLike, rx: re.Pattern, value, mask: npt.NDArray[np.bool_] | None) -> None:
    """
    Parameters
    ----------
    values : ArrayLike
        Object dtype.
    rx : re.Pattern
    value : Any
    mask : np.ndarray[bool], optional

    Notes
    -----
    Alters values in-place.
    """
    if isna(value) or not isinstance(value, str):

        def re_replacer(s):
            if is_re(rx) and isinstance(s, str):
                return value if rx.search(s) is not None else s
            else:
                return s
    else:

        def re_replacer(s):
            if is_re(rx) and isinstance(s, str):
                return rx.sub(value, s)
            else:
                return s
    f = np.vectorize(re_replacer, otypes=[np.object_])
    if mask is None:
        values[:] = f(values)
    else:
        values[mask] = f(values[mask])