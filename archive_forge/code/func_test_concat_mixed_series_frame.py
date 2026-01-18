import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_concat_mixed_series_frame(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'c': 1})
    ser = Series([4, 5, 6], name='d')
    result = concat([df, ser], axis=1)
    expected = result.copy()
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
        assert np.shares_memory(get_array(result, 'c'), get_array(df, 'c'))
        assert np.shares_memory(get_array(result, 'd'), get_array(ser, 'd'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
        assert not np.shares_memory(get_array(result, 'c'), get_array(df, 'c'))
        assert not np.shares_memory(get_array(result, 'd'), get_array(ser, 'd'))
    ser.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'd'), get_array(ser, 'd'))
    df.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(result, expected)