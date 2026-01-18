import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_loc_empty_multiindex():
    arrays = [['a', 'a', 'b', 'a'], ['a', 'a', 'b', 'b']]
    index = MultiIndex.from_arrays(arrays, names=('idx1', 'idx2'))
    df = DataFrame([1, 2, 3, 4], index=index, columns=['value'])
    empty_multiindex = df.loc[df.loc[:, 'value'] == 0, :].index
    result = df.loc[empty_multiindex, :]
    expected = df.loc[[False] * len(df.index), :]
    tm.assert_frame_equal(result, expected)
    df.loc[df.loc[df.loc[:, 'value'] == 0].index, 'value'] = 5
    result = df
    expected = DataFrame([1, 2, 3, 4], index=index, columns=['value'])
    tm.assert_frame_equal(result, expected)