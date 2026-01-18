import pytest
from pandas import interval_range
import pandas._testing as tm
@pytest.mark.parametrize('kwargs', [{'start': 0, 'periods': 4}, {'start': 1, 'periods': 5}, {'start': 5, 'end': 10, 'closed': 'left'}])
def test_interval_array_equal(kwargs):
    arr = interval_range(**kwargs).values
    tm.assert_interval_array_equal(arr, arr)