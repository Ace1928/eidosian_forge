import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_del_frame(backend, using_copy_on_write, warn_copy_on_write):
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df[:]
    assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    del df2['b']
    assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    tm.assert_frame_equal(df, df_orig)
    tm.assert_frame_equal(df2, df_orig[['a', 'c']])
    df2._mgr._verify_integrity()
    with tm.assert_cow_warning(warn_copy_on_write and dtype_backend == 'numpy'):
        df.loc[0, 'b'] = 200
    assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    df_orig = df.copy()
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.loc[0, 'a'] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.loc[0, 'a'] == 100