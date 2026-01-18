from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_cache_updating2(self, using_copy_on_write):
    df = DataFrame(np.zeros((5, 5), dtype='int64'), columns=['a', 'b', 'c', 'd', 'e'], index=range(5))
    df['f'] = 0
    df_orig = df.copy()
    if using_copy_on_write:
        with pytest.raises(ValueError, match='read-only'):
            df.f.values[3] = 1
        tm.assert_frame_equal(df, df_orig)
        return
    df.f.values[3] = 1
    df.f.values[3] = 2
    expected = DataFrame(np.zeros((5, 6), dtype='int64'), columns=['a', 'b', 'c', 'd', 'e', 'f'], index=range(5))
    expected.at[3, 'f'] = 2
    tm.assert_frame_equal(df, expected)
    expected = Series([0, 0, 0, 2, 0], name='f')
    tm.assert_series_equal(df.f, expected)