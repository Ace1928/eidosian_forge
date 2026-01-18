import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_dict(using_copy_on_write):
    df = DataFrame({'a': [1.5, np.nan], 'b': 1})
    df_orig = df.copy()
    df2 = df.fillna({'a': 100.5})
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    else:
        assert not np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
    df2.iloc[0, 1] = 100
    tm.assert_frame_equal(df_orig, df)