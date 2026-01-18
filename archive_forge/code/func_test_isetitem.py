import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_isetitem(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df_orig = df.copy()
    df2 = df.copy(deep=None)
    df2.isetitem(1, np.array([-1, -2, -3]))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
        assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    else:
        assert not np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    df2.loc[0, 'a'] = 0
    tm.assert_frame_equal(df, df_orig)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
    else:
        assert not np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))