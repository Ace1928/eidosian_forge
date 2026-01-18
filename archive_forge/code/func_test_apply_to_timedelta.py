import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_to_timedelta(by_row):
    list_of_valid_strings = ['00:00:01', '00:00:02']
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)
    list_of_strings = ['00:00:01', np.nan, pd.NaT, pd.NaT]
    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)