import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('func', ['max', 'min'])
def test_groupby_rolling_index_changed(self, func):
    ds = Series([1, 2, 2], index=MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('c', 'z')], names=['1', '2']), name='a')
    result = getattr(ds.groupby(ds).rolling(2), func)()
    expected = Series([np.nan, np.nan, 2.0], index=MultiIndex.from_tuples([(1, 'a', 'x'), (2, 'a', 'y'), (2, 'c', 'z')], names=['a', '1', '2']), name='a')
    tm.assert_series_equal(result, expected)