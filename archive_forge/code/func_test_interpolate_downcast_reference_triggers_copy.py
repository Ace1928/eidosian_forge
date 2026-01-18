import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_interpolate_downcast_reference_triggers_copy(using_copy_on_write):
    df = DataFrame({'a': [1, np.nan, 2.5], 'b': 1})
    df_orig = df.copy()
    arr_a = get_array(df, 'a')
    view = df[:]
    msg = 'DataFrame.interpolate with method=pad is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.interpolate(method='pad', inplace=True, downcast='infer')
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
        assert not np.shares_memory(arr_a, get_array(df, 'a'))
        tm.assert_frame_equal(df_orig, view)
    else:
        tm.assert_frame_equal(df, view)