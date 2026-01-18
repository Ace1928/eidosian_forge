import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.filterwarnings('ignore:All-NaN slice encountered:RuntimeWarning')
def test_median_2d(self, arr1d):
    arr = arr1d.reshape(1, -1)
    assert arr.median() == arr1d.median()
    assert arr.median(skipna=False) is NaT
    result = arr.median(axis=0)
    expected = arr1d
    tm.assert_equal(result, expected)
    result = arr.median(axis=0, skipna=False)
    expected = arr1d
    tm.assert_equal(result, expected)
    result = arr.median(axis=1)
    expected = type(arr)._from_sequence([arr1d.median()], dtype=arr.dtype)
    tm.assert_equal(result, expected)
    result = arr.median(axis=1, skipna=False)
    expected = type(arr)._from_sequence([NaT], dtype=arr.dtype)
    tm.assert_equal(result, expected)