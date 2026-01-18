import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_df(self):
    df1 = DataFrame({'A': [1, 2, 3, 4], 'B': [2, np.nan, 4, 4]})
    df2 = DataFrame({'A': [0, 2, 12, 4], 'B': [2, np.nan, 4, 5]})
    expected = DataFrame(False, df1.index, df1.columns)
    result = df1.isin(df2)
    expected.loc[[1, 3], 'A'] = True
    expected.loc[[0, 2], 'B'] = True
    tm.assert_frame_equal(result, expected)
    df2.columns = ['A', 'C']
    result = df1.isin(df2)
    expected['B'] = False
    tm.assert_frame_equal(result, expected)