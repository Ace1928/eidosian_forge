import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_on_tz_aware_datetimeindex(self):
    df1 = DataFrame({'date': pd.date_range(start='2018-01-01', periods=5, tz='America/Chicago'), 'vals': list('abcde')})
    df2 = DataFrame({'date': pd.date_range(start='2018-01-03', periods=5, tz='America/Chicago'), 'vals_2': list('tuvwx')})
    result = df1.join(df2.set_index('date'), on='date')
    expected = df1.copy()
    expected['vals_2'] = Series([np.nan] * 2 + list('tuv'))
    tm.assert_frame_equal(result, expected)