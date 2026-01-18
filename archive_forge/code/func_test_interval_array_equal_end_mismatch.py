import pytest
from pandas import interval_range
import pandas._testing as tm
def test_interval_array_equal_end_mismatch():
    kwargs = {'start': 0, 'periods': 5}
    arr1 = interval_range(end=10, **kwargs).values
    arr2 = interval_range(end=20, **kwargs).values
    msg = 'IntervalArray.left are different\n\nIntervalArray.left values are different \\(80.0 %\\)\n\\[left\\]:  \\[0, 2, 4, 6, 8\\]\n\\[right\\]: \\[0, 4, 8, 12, 16\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)