import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_empty_tz_frame_with_datetime64ns(self, using_array_manager):
    df = DataFrame(columns=['a']).astype('datetime64[ns, UTC]')
    result = df._append({'a': pd.NaT}, ignore_index=True)
    if using_array_manager:
        expected = DataFrame({'a': [pd.NaT]}, dtype=object)
    else:
        expected = DataFrame({'a': [np.nan]}, dtype=object)
    tm.assert_frame_equal(result, expected)
    df = DataFrame(columns=['a']).astype('datetime64[ns, UTC]')
    other = Series({'a': pd.NaT}, dtype='datetime64[ns]')
    result = df._append(other, ignore_index=True)
    tm.assert_frame_equal(result, expected)
    other = Series({'a': pd.NaT}, dtype='datetime64[ns, US/Pacific]')
    result = df._append(other, ignore_index=True)
    expected = DataFrame({'a': [pd.NaT]}).astype(object)
    tm.assert_frame_equal(result, expected)