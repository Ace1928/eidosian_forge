import decimal
import numpy as np
import pytest
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('klass', [np.datetime64, np.timedelta64])
def test_datetime_likes_nan(klass):
    dtype = klass.__name__ + '[ns]'
    arr = np.array([1, 2, np.nan])
    exp = np.array([1, 2, klass('NaT')], dtype)
    res = maybe_downcast_to_dtype(arr, dtype)
    tm.assert_numpy_array_equal(res, exp)