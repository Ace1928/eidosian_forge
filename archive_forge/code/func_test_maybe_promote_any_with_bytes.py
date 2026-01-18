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
def test_maybe_promote_any_with_bytes(any_numpy_dtype):
    dtype = np.dtype(any_numpy_dtype)
    fill_value = b'abc'
    expected_dtype = np.dtype(np.object_)
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)