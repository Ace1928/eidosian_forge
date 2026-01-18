import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index, expected_index', [(['w', 'x', 'y', 'z'], ['w', 'w', 'y', 'y']), ([3, 4, 5, 6], [3, 3, 5, 5])])
def test_groupby_sample_with_weights(index, expected_index):
    values = [1] * 2 + [2] * 2
    df = DataFrame({'a': values, 'b': values}, index=Index(index))
    result = df.groupby('a').sample(n=2, replace=True, weights=[1, 0, 1, 0])
    expected = DataFrame({'a': values, 'b': values}, index=Index(expected_index))
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a')['b'].sample(n=2, replace=True, weights=[1, 0, 1, 0])
    expected = Series(values, name='b', index=Index(expected_index))
    tm.assert_series_equal(result, expected)