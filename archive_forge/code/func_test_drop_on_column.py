import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_drop_on_column(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.drop(columns='a')
    df2._mgr._verify_integrity()
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'b'), get_array(df, 'b'))
        assert np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
    else:
        assert not np.shares_memory(get_array(df2, 'b'), get_array(df, 'b'))
        assert not np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, 'b'), get_array(df, 'b'))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
    tm.assert_frame_equal(df, df_orig)