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
@pytest.mark.parametrize('fill_value', [pd.Timedelta(days=1), np.timedelta64(24, 'h'), datetime.timedelta(1)], ids=['pd.Timedelta', 'np.timedelta64', 'datetime.timedelta'])
def test_maybe_promote_any_with_timedelta64(any_numpy_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype)
    if dtype.kind == 'm':
        expected_dtype = dtype
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)