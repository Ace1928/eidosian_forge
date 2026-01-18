import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_fill_consistency():
    df = DataFrame(index=pd.MultiIndex.from_product([['value1', 'value2'], date_range('2014-01-01', '2014-01-06')]), columns=Index(['1', '2'], name='id'))
    df['1'] = [np.nan, 1, np.nan, np.nan, 11, np.nan, np.nan, 2, np.nan, np.nan, 22, np.nan]
    df['2'] = [np.nan, 3, np.nan, np.nan, 33, np.nan, np.nan, 4, np.nan, np.nan, 44, np.nan]
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby(level=0, axis=0).fillna(method='ffill')
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.T.groupby(level=0, axis=1).fillna(method='ffill').T
    tm.assert_frame_equal(result, expected)