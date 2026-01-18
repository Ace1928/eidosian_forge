import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_astype_dict_dtypes(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': Series([1.5, 1.5, 1.5], dtype='float64')})
    df_orig = df.copy()
    df2 = df.astype({'a': 'float64', 'c': 'float64'})
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
        assert np.shares_memory(get_array(df2, 'b'), get_array(df, 'b'))
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
        assert not np.shares_memory(get_array(df2, 'b'), get_array(df, 'b'))
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 2] = 5.5
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, 'c'), get_array(df, 'c'))
    df2.iloc[0, 1] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, 'b'), get_array(df, 'b'))
    tm.assert_frame_equal(df, df_orig)