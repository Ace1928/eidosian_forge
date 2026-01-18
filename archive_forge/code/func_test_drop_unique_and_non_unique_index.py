import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_bool_dtype
@pytest.mark.parametrize('data, index, drop_labels, axis, expected_data, expected_index', [([1, 2], ['one', 'two'], ['two'], 0, [1], ['one']), ([1, 2], ['one', 'two'], ['two'], 'rows', [1], ['one']), ([1, 1, 2], ['one', 'two', 'one'], ['two'], 0, [1, 2], ['one', 'one']), ([1, 1, 2], ['one', 'two', 'one'], 'two', 0, [1, 2], ['one', 'one']), ([1, 1, 2], ['one', 'two', 'one'], ['one'], 0, [1], ['two']), ([1, 1, 2], ['one', 'two', 'one'], 'one', 0, [1], ['two'])])
def test_drop_unique_and_non_unique_index(data, index, axis, drop_labels, expected_data, expected_index):
    ser = Series(data=data, index=index)
    result = ser.drop(drop_labels, axis=axis)
    expected = Series(data=expected_data, index=expected_index)
    tm.assert_series_equal(result, expected)