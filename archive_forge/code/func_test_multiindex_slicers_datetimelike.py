from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_multiindex_slicers_datetimelike(self):
    dates = [datetime(2012, 1, 1, 12, 12, 12) + timedelta(days=i) for i in range(6)]
    freq = [1, 2]
    index = MultiIndex.from_product([dates, freq], names=['date', 'frequency'])
    df = DataFrame(np.arange(6 * 2 * 4, dtype='int64').reshape(-1, 4), index=index, columns=list('ABCD'))
    idx = pd.IndexSlice
    expected = df.iloc[[0, 2, 4], [0, 1]]
    result = df.loc[(slice(Timestamp('2012-01-01 12:12:12'), Timestamp('2012-01-03 12:12:12')), slice(1, 1)), slice('A', 'B')]
    tm.assert_frame_equal(result, expected)
    result = df.loc[(idx[Timestamp('2012-01-01 12:12:12'):Timestamp('2012-01-03 12:12:12')], idx[1:1]), slice('A', 'B')]
    tm.assert_frame_equal(result, expected)
    result = df.loc[(slice(Timestamp('2012-01-01 12:12:12'), Timestamp('2012-01-03 12:12:12')), 1), slice('A', 'B')]
    tm.assert_frame_equal(result, expected)
    result = df.loc[(slice('2012-01-01 12:12:12', '2012-01-03 12:12:12'), slice(1, 1)), slice('A', 'B')]
    tm.assert_frame_equal(result, expected)
    result = df.loc[(idx['2012-01-01 12:12:12':'2012-01-03 12:12:12'], 1), idx['A', 'B']]
    tm.assert_frame_equal(result, expected)