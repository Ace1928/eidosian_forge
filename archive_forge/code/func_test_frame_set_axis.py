import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_frame_set_axis(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.set_axis(['a', 'b', 'c'], axis='index')
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)