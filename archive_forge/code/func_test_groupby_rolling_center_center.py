import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_center_center(self):
    series = Series(range(1, 6))
    result = series.groupby(series).rolling(center=True, window=3).mean()
    expected = Series([np.nan] * 5, index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))))
    tm.assert_series_equal(result, expected)
    series = Series(range(1, 5))
    result = series.groupby(series).rolling(center=True, window=3).mean()
    expected = Series([np.nan] * 4, index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))))
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': ['a'] * 5 + ['b'] * 6, 'b': range(11)})
    result = df.groupby('a').rolling(center=True, window=3).mean()
    expected = DataFrame([np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan], index=MultiIndex.from_tuples((('a', 0), ('a', 1), ('a', 2), ('a', 3), ('a', 4), ('b', 5), ('b', 6), ('b', 7), ('b', 8), ('b', 9), ('b', 10)), names=['a', None]), columns=['b'])
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'a': ['a'] * 5 + ['b'] * 5, 'b': range(10)})
    result = df.groupby('a').rolling(center=True, window=3).mean()
    expected = DataFrame([np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan], index=MultiIndex.from_tuples((('a', 0), ('a', 1), ('a', 2), ('a', 3), ('a', 4), ('b', 5), ('b', 6), ('b', 7), ('b', 8), ('b', 9)), names=['a', None]), columns=['b'])
    tm.assert_frame_equal(result, expected)