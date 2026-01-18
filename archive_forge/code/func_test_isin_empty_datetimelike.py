import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_empty_datetimelike(self):
    df1_ts = DataFrame({'date': pd.to_datetime(['2014-01-01', '2014-01-02'])})
    df1_td = DataFrame({'date': [pd.Timedelta(1, 's'), pd.Timedelta(2, 's')]})
    df2 = DataFrame({'date': []})
    df3 = DataFrame()
    expected = DataFrame({'date': [False, False]})
    result = df1_ts.isin(df2)
    tm.assert_frame_equal(result, expected)
    result = df1_ts.isin(df3)
    tm.assert_frame_equal(result, expected)
    result = df1_td.isin(df2)
    tm.assert_frame_equal(result, expected)
    result = df1_td.isin(df3)
    tm.assert_frame_equal(result, expected)