from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_nan_is_false():
    df = DataFrame({'A': np.arange(8), 'B': list('aabbbbcc'), 'C': np.arange(8)})
    s = df['B']
    g_df = df.groupby(df['B'])
    g_s = s.groupby(s)
    f = lambda x: np.nan
    tm.assert_frame_equal(g_df.filter(f), df.loc[[]])
    tm.assert_series_equal(g_s.filter(f), s[[]])