import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_align_copy_false(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df_orig = df.copy()
    df2, df3 = df.align(df, copy=False)
    assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
    assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    if using_copy_on_write:
        df2.loc[0, 'a'] = 0
        tm.assert_frame_equal(df, df_orig)
        df3.loc[0, 'a'] = 0
        tm.assert_frame_equal(df, df_orig)