import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_resulting_multiindex2(self):
    df = DataFrame({'a': np.arange(12.0), 'b': [1, 2] * 6, 'c': [1, 2, 3, 4] * 3})
    result = df.groupby(['b', 'c']).rolling(2).sum()
    expected_index = MultiIndex.from_tuples([(1, 1, 0), (1, 1, 4), (1, 1, 8), (1, 3, 2), (1, 3, 6), (1, 3, 10), (2, 2, 1), (2, 2, 5), (2, 2, 9), (2, 4, 3), (2, 4, 7), (2, 4, 11)], names=['b', 'c', None])
    tm.assert_index_equal(result.index, expected_index)