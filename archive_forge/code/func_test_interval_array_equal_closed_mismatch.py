import pytest
from pandas import interval_range
import pandas._testing as tm
def test_interval_array_equal_closed_mismatch():
    kwargs = {'start': 0, 'periods': 5}
    arr1 = interval_range(closed='left', **kwargs).values
    arr2 = interval_range(closed='right', **kwargs).values
    msg = 'IntervalArray are different\n\nAttribute "closed" are different\n\\[left\\]:  left\n\\[right\\]: right'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)