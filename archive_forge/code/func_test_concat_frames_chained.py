import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_concat_frames_chained(using_copy_on_write):
    df1 = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    df2 = DataFrame({'c': [4, 5, 6]})
    df3 = DataFrame({'d': [4, 5, 6]})
    result = concat([concat([df1, df2], axis=1), df3], axis=1)
    expected = result.copy()
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert np.shares_memory(get_array(result, 'c'), get_array(df2, 'c'))
        assert np.shares_memory(get_array(result, 'd'), get_array(df3, 'd'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert not np.shares_memory(get_array(result, 'c'), get_array(df2, 'c'))
        assert not np.shares_memory(get_array(result, 'd'), get_array(df3, 'd'))
    df1.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
    tm.assert_frame_equal(result, expected)