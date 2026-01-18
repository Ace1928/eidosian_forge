import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_zerolen_series_columns_align(self):
    df = DataFrame(columns=['A', 'B'])
    df.loc[0] = Series(1, index=range(4))
    expected = DataFrame(columns=['A', 'B'], index=[0], dtype=np.float64)
    tm.assert_frame_equal(df, expected)
    df = DataFrame(columns=['A', 'B'])
    df.loc[0] = Series(1, index=['B'])
    exp = DataFrame([[np.nan, 1]], columns=['A', 'B'], index=[0], dtype='float64')
    tm.assert_frame_equal(df, exp)