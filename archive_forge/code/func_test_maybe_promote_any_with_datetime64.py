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
@pytest.mark.parametrize('fill_value', [pd.Timestamp('now'), np.datetime64('now'), datetime.datetime.now(), datetime.date.today()], ids=['pd.Timestamp', 'np.datetime64', 'datetime.datetime', 'datetime.date'])
def test_maybe_promote_any_with_datetime64(any_numpy_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype)
    if dtype.kind == 'M':
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    if type(fill_value) is datetime.date and dtype.kind == 'M':
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)