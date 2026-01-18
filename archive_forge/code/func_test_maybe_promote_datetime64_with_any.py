import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs import NaT
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
import pandas as pd
def test_maybe_promote_datetime64_with_any(datetime64_dtype, any_numpy_dtype):
    dtype = np.dtype(datetime64_dtype)
    fill_dtype = np.dtype(any_numpy_dtype)
    fill_value = np.array([1], dtype=fill_dtype)[0]
    if fill_dtype.kind == 'M':
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)