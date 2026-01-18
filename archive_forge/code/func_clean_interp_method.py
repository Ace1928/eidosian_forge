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
def clean_interp_method(method: str, index: Index, **kwargs) -> str:
    order = kwargs.get('order')
    if method in ('spline', 'polynomial') and order is None:
        raise ValueError('You must specify the order of the spline or polynomial.')
    valid = NP_METHODS + SP_METHODS
    if method not in valid:
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")
    if method in ('krogh', 'piecewise_polynomial', 'pchip'):
        if not index.is_monotonic_increasing:
            raise ValueError(f'{method} interpolation requires that the index be monotonic.')
    return method