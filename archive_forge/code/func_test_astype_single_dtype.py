import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_astype_single_dtype(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': 1.5})
    df_orig = df.copy()
    df2 = df.astype('float64')
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 2] = 5.5
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
    tm.assert_frame_equal(df, df_orig)
    df2 = df.astype('float64')
    df.iloc[0, 2] = 5.5
    tm.assert_frame_equal(df2, df_orig.astype('float64'))