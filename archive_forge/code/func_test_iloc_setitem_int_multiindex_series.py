import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, indexes, values, expected_k', [([[2, 22, 5], [2, 33, 6]], [0, -1, 1], [2, 3, 1], [7, 10]), ([[1, 22, 555], [1, 33, 666]], [0, -1, 1], [200, 300, 100], [755, 1066]), ([[1, 3, 7], [2, 4, 8]], [0, -1, 1], [10, 10, 1000], [17, 1018]), ([[1, 11, 4], [2, 22, 5], [3, 33, 6]], [0, -1, 1], [4, 7, 10], [8, 15, 13])])
def test_iloc_setitem_int_multiindex_series(data, indexes, values, expected_k):
    df = DataFrame(data=data, columns=['i', 'j', 'k'])
    df = df.set_index(['i', 'j'])
    series = df.k.copy()
    for i, v in zip(indexes, values):
        series.iloc[i] += v
    df['k'] = expected_k
    expected = df.k
    tm.assert_series_equal(series, expected)