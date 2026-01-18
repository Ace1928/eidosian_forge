import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
@pytest.mark.parametrize('dtype', [Int8Dtype(), 'Int8', UInt32Dtype(), 'UInt32'])
def test_astype_specific_casting(dtype):
    s = pd.Series([1, 2, 3], dtype='Int64')
    result = s.astype(dtype)
    expected = pd.Series([1, 2, 3], dtype=dtype)
    tm.assert_series_equal(result, expected)
    s = pd.Series([1, 2, 3, None], dtype='Int64')
    result = s.astype(dtype)
    expected = pd.Series([1, 2, 3, None], dtype=dtype)
    tm.assert_series_equal(result, expected)