import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_assert_series_equal_interval_dtype_mismatch():
    left = Series([pd.Interval(0, 1)], dtype='interval')
    right = left.astype(object)
    msg = 'Attributes of Series are different\n\nAttribute "dtype" are different\n\\[left\\]:  interval\\[int64, right\\]\n\\[right\\]: object'
    tm.assert_series_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)