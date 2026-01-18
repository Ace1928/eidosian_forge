import pytest
from pandas import interval_range
import pandas._testing as tm
def test_interval_array_equal_periods_mismatch():
    kwargs = {'start': 0}
    arr1 = interval_range(periods=5, **kwargs).values
    arr2 = interval_range(periods=6, **kwargs).values
    msg = 'IntervalArray.left are different\n\nIntervalArray.left shapes are different\n\\[left\\]:  \\(5,\\)\n\\[right\\]: \\(6,\\)'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)