import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('df_columns, series_index', combinations(indexes_can_append, r=2), ids=lambda x: type(x).__name__)
def test_append_different_columns_types(self, df_columns, series_index):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=df_columns)
    ser = Series([7, 8, 9], index=series_index, name=2)
    result = df._append(ser)
    idx_diff = ser.index.difference(df_columns)
    combined_columns = Index(df_columns.tolist()).append(idx_diff)
    expected = DataFrame([[1.0, 2.0, 3.0, np.nan, np.nan, np.nan], [4, 5, 6, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, 7, 8, 9]], index=[0, 1, 2], columns=combined_columns)
    tm.assert_frame_equal(result, expected)