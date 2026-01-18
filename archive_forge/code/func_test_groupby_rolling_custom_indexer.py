import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_custom_indexer(self):

    class SimpleIndexer(BaseIndexer):

        def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None, step=None):
            min_periods = self.window_size if min_periods is None else 0
            end = np.arange(num_values, dtype=np.int64) + 1
            start = end.copy() - self.window_size
            start[start < 0] = min_periods
            return (start, end)
    df = DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5)
    result = df.groupby(df.index).rolling(SimpleIndexer(window_size=3), min_periods=1).sum()
    expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
    tm.assert_frame_equal(result, expected)