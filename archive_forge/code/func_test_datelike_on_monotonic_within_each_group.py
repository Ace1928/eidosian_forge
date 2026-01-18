import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_datelike_on_monotonic_within_each_group(self):
    dates = date_range(start='2016-01-01 09:30:00', periods=20, freq='s')
    df = DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.concatenate((dates, dates)), 'C': np.arange(40)})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.set_index('B').groupby('A').apply(lambda x: x.rolling('4s')['C'].mean())
    result = df.groupby('A').rolling('4s', on='B').C.mean()
    tm.assert_series_equal(result, expected)