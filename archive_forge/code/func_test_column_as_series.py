import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_column_as_series(backend, using_copy_on_write, warn_copy_on_write, using_array_manager):
    dtype_backend, DataFrame, Series = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    s = df['a']
    assert np.shares_memory(get_array(s, 'a'), get_array(df, 'a'))
    if using_copy_on_write or using_array_manager:
        s[0] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            s[0] = 0
    else:
        warn = SettingWithCopyWarning if dtype_backend == 'numpy' else None
        with pd.option_context('chained_assignment', 'warn'):
            with tm.assert_produces_warning(warn):
                s[0] = 0
    expected = Series([0, 2, 3], name='a')
    tm.assert_series_equal(s, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
        tm.assert_series_equal(df['a'], df_orig['a'])
    else:
        df_orig.iloc[0, 0] = 0
        tm.assert_frame_equal(df, df_orig)