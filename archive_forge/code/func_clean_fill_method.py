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
def clean_fill_method(method: Literal['ffill', 'pad', 'bfill', 'backfill', 'nearest'], *, allow_nearest: bool=False) -> Literal['pad', 'backfill', 'nearest']:
    if isinstance(method, str):
        method = method.lower()
        if method == 'ffill':
            method = 'pad'
        elif method == 'bfill':
            method = 'backfill'
    valid_methods = ['pad', 'backfill']
    expecting = 'pad (ffill) or backfill (bfill)'
    if allow_nearest:
        valid_methods.append('nearest')
        expecting = 'pad (ffill), backfill (bfill) or nearest'
    if method not in valid_methods:
        raise ValueError(f'Invalid fill method. Expecting {expecting}. Got {method}')
    return method