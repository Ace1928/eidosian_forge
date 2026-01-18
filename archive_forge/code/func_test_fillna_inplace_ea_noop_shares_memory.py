import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_inplace_ea_noop_shares_memory(using_copy_on_write, warn_copy_on_write, any_numeric_ea_and_arrow_dtype):
    df = DataFrame({'a': [1, NA, 3], 'b': 1}, dtype=any_numeric_ea_and_arrow_dtype)
    df_orig = df.copy()
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.fillna(100, inplace=True)
    if isinstance(df['a'].dtype, ArrowDtype) or using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), get_array(view, 'a'))
    else:
        assert np.shares_memory(get_array(df, 'a'), get_array(view, 'a'))
    assert np.shares_memory(get_array(df, 'b'), get_array(view, 'b'))
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(1)
        assert not view._mgr._has_no_reference(1)
    with tm.assert_cow_warning(warn_copy_on_write and 'pyarrow' not in any_numeric_ea_and_arrow_dtype):
        df.iloc[0, 1] = 100
    if isinstance(df['a'].dtype, ArrowDtype) or using_copy_on_write:
        tm.assert_frame_equal(df_orig, view)
    else:
        tm.assert_frame_equal(df, view)