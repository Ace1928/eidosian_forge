from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def is_any_real_numeric_dtype(arr_or_dtype) -> bool:
    try:
        return pd.api.types.is_any_real_numeric_dtype(arr_or_dtype)
    except AttributeError:
        from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
        return is_numeric_dtype(arr_or_dtype) and (not is_complex_dtype(arr_or_dtype)) and (not is_bool_dtype(arr_or_dtype))