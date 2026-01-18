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
def get_fill_func(method, ndim: int=1):
    method = clean_fill_method(method)
    if ndim == 1:
        return _fill_methods[method]
    return {'pad': _pad_2d, 'backfill': _backfill_2d}[method]