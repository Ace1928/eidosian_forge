import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_list_of_tuples_with_multiindex(self, multiindex_year_month_day_dataframe_random_data):
    ser = multiindex_year_month_day_dataframe_random_data['A']
    expected = ser.reindex(ser.index[49:51])
    result = ser.loc[[(2000, 3, 10), (2000, 3, 13)]]
    tm.assert_series_equal(result, expected)