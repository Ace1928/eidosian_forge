import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_fancy_slice_partial(self, multiindex_dataframe_random_data, multiindex_year_month_day_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    result = frame.loc['bar':'baz']
    expected = frame[3:7]
    tm.assert_frame_equal(result, expected)
    ymd = multiindex_year_month_day_dataframe_random_data
    result = ymd.loc[(2000, 2):(2000, 4)]
    lev = ymd.index.codes[1]
    expected = ymd[(lev >= 1) & (lev <= 3)]
    tm.assert_frame_equal(result, expected)