import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', ['where', 'mask'])
def test_masking_inplace(using_copy_on_write, method, warn_copy_on_write):
    df = DataFrame({'a': [1.5, 2, 3]})
    df_orig = df.copy()
    arr_a = get_array(df, 'a')
    view = df[:]
    method = getattr(df, method)
    if warn_copy_on_write:
        with tm.assert_cow_warning():
            method(df['a'] > 1.6, -1, inplace=True)
    else:
        method(df['a'] > 1.6, -1, inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), arr_a)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'a'), arr_a)