import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_interpolate_cleaned_fill_method(using_copy_on_write):
    df = DataFrame({'a': ['a', np.nan, 'c'], 'b': 1})
    df_orig = df.copy()
    msg = 'DataFrame.interpolate with object dtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.interpolate(method='linear')
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    result.iloc[0, 0] = Timestamp('2021-12-31')
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)