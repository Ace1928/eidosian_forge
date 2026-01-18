import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_column_as_series_set_with_upcast(backend, using_copy_on_write, using_array_manager, warn_copy_on_write):
    dtype_backend, DataFrame, Series = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    s = df['a']
    if dtype_backend == 'nullable':
        with tm.assert_cow_warning(warn_copy_on_write):
            with pytest.raises(TypeError, match='Invalid value'):
                s[0] = 'foo'
        expected = Series([1, 2, 3], name='a')
    elif using_copy_on_write or warn_copy_on_write or using_array_manager:
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            s[0] = 'foo'
        expected = Series(['foo', 2, 3], dtype=object, name='a')
    else:
        with pd.option_context('chained_assignment', 'warn'):
            msg = '|'.join(['A value is trying to be set on a copy of a slice from a DataFrame', 'Setting an item of incompatible dtype is deprecated'])
            with tm.assert_produces_warning((SettingWithCopyWarning, FutureWarning), match=msg):
                s[0] = 'foo'
        expected = Series(['foo', 2, 3], dtype=object, name='a')
    tm.assert_series_equal(s, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
        tm.assert_series_equal(df['a'], df_orig['a'])
    else:
        df_orig['a'] = expected
        tm.assert_frame_equal(df, df_orig)