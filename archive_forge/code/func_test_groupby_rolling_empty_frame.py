import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_empty_frame(self):
    expected = DataFrame({'s1': []})
    result = expected.groupby('s1').rolling(window=1).sum()
    expected = expected.drop(columns='s1')
    expected.index = MultiIndex.from_product([Index([], dtype='float64'), Index([], dtype='int64')], names=['s1', None])
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'s1': [], 's2': []})
    result = expected.groupby(['s1', 's2']).rolling(window=1).sum()
    expected = expected.drop(columns=['s1', 's2'])
    expected.index = MultiIndex.from_product([Index([], dtype='float64'), Index([], dtype='float64'), Index([], dtype='int64')], names=['s1', 's2', None])
    tm.assert_frame_equal(result, expected)