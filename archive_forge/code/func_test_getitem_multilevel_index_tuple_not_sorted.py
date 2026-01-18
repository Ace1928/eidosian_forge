import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_getitem_multilevel_index_tuple_not_sorted(self):
    index_columns = list('abc')
    df = DataFrame([[0, 1, 0, 'x'], [0, 0, 1, 'y']], columns=index_columns + ['data'])
    df = df.set_index(index_columns)
    query_index = df.index[:1]
    rs = df.loc[query_index, 'data']
    xp_idx = MultiIndex.from_tuples([(0, 1, 0)], names=['a', 'b', 'c'])
    xp = Series(['x'], index=xp_idx, name='data')
    tm.assert_series_equal(rs, xp)