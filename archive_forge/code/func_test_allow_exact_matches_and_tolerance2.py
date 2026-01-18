import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_allow_exact_matches_and_tolerance2(self):
    df1 = pd.DataFrame({'time': to_datetime(['2016-07-15 13:30:00.030']), 'username': ['bob']})
    df2 = pd.DataFrame({'time': to_datetime(['2016-07-15 13:30:00.000', '2016-07-15 13:30:00.030']), 'version': [1, 2]})
    result = merge_asof(df1, df2, on='time')
    expected = pd.DataFrame({'time': to_datetime(['2016-07-15 13:30:00.030']), 'username': ['bob'], 'version': [2]})
    tm.assert_frame_equal(result, expected)
    result = merge_asof(df1, df2, on='time', allow_exact_matches=False)
    expected = pd.DataFrame({'time': to_datetime(['2016-07-15 13:30:00.030']), 'username': ['bob'], 'version': [1]})
    tm.assert_frame_equal(result, expected)
    result = merge_asof(df1, df2, on='time', allow_exact_matches=False, tolerance=Timedelta('10ms'))
    expected = pd.DataFrame({'time': to_datetime(['2016-07-15 13:30:00.030']), 'username': ['bob'], 'version': [np.nan]})
    tm.assert_frame_equal(result, expected)