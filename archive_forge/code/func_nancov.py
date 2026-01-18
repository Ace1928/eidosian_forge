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
def nancov(a: np.ndarray, b: np.ndarray, *, min_periods: int | None=None, ddof: int | None=1) -> float:
    if len(a) != len(b):
        raise AssertionError('Operands to nancov must have same size')
    if min_periods is None:
        min_periods = 1
    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]
    if len(a) < min_periods:
        return np.nan
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)
    return np.cov(a, b, ddof=ddof)[0, 1]