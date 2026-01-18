from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def is_string_dtype(arr_or_dtype) -> bool:
    if hasattr(arr_or_dtype, 'dtype'):
        dtype = arr_or_dtype.dtype
    else:
        dtype = arr_or_dtype
    if not PANDAS_GE_200:
        return pd.api.types.is_dtype_equal(dtype, 'string')
    return pd.api.types.is_string_dtype(dtype)