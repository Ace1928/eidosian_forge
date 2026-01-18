import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_tuple_plus_columns(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    df = ymd[:5]
    result = df.loc[(2000, 1, 6), ['A', 'B', 'C']]
    expected = df.loc[2000, 1, 6][['A', 'B', 'C']]
    tm.assert_series_equal(result, expected)