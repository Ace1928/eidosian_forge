import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_eval_inplace(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': 1})
    df_orig = df.copy()
    df_view = df[:]
    df.eval('c = a+b', inplace=True)
    assert np.shares_memory(get_array(df, 'a'), get_array(df_view, 'a'))
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df_view, df_orig)