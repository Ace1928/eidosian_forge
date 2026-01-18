import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', ['pad', 'nearest', 'linear'])
def test_interpolate_no_op(using_copy_on_write, method):
    df = DataFrame({'a': [1, 2]})
    df_orig = df.copy()
    warn = None
    if method == 'pad':
        warn = FutureWarning
    msg = 'DataFrame.interpolate with method=pad is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        result = df.interpolate(method=method)
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    result.iloc[0, 0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)