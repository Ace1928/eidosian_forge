import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_frame_getitem_not_sorted(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    df = frame.T
    df['foo', 'four'] = 'foo'
    arrays = [np.array(x) for x in zip(*df.columns.values)]
    result = df['foo']
    result2 = df.loc[:, 'foo']
    expected = df.reindex(columns=df.columns[arrays[0] == 'foo'])
    expected.columns = expected.columns.droplevel(0)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)
    df = df.T
    result = df.xs('foo')
    result2 = df.loc['foo']
    expected = df.reindex(df.index[arrays[0] == 'foo'])
    expected.index = expected.index.droplevel(0)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected)