import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('use_mgr', [True, False])
@pytest.mark.parametrize('columns', [None, ['a']])
def test_dataframe_constructor_mgr_or_df(using_copy_on_write, warn_copy_on_write, columns, use_mgr):
    df = DataFrame({'a': [1, 2, 3]})
    df_orig = df.copy()
    if use_mgr:
        data = df._mgr
        warn = DeprecationWarning
    else:
        data = df
        warn = None
    msg = 'Passing a BlockManager to DataFrame'
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        new_df = DataFrame(data)
    assert np.shares_memory(get_array(df, 'a'), get_array(new_df, 'a'))
    with tm.assert_cow_warning(warn_copy_on_write and (not use_mgr)):
        new_df.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), get_array(new_df, 'a'))
        tm.assert_frame_equal(df, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'a'), get_array(new_df, 'a'))
        tm.assert_frame_equal(df, new_df)