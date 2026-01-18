import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('vals', [[1, np.nan, 2], [Timestamp('2019-12-31'), NaT, Timestamp('2020-12-31')]])
def test_interpolate_inplace_with_refs(using_copy_on_write, vals, warn_copy_on_write):
    df = DataFrame({'a': [1, np.nan, 2]})
    df_orig = df.copy()
    arr = get_array(df, 'a')
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.interpolate(method='linear', inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(arr, get_array(df, 'a'))
        tm.assert_frame_equal(df_orig, view)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
    else:
        assert np.shares_memory(arr, get_array(df, 'a'))