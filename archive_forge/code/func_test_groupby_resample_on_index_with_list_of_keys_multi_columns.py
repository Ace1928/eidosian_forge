from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_groupby_resample_on_index_with_list_of_keys_multi_columns():
    df = DataFrame(data={'group': [0, 0, 0, 0, 1, 1, 1, 1], 'first_val': [3, 1, 4, 1, 5, 9, 2, 6], 'second_val': [2, 7, 1, 8, 2, 8, 1, 8], 'third_val': [1, 4, 1, 4, 2, 1, 3, 5]}, index=date_range(start='2016-01-01', periods=8, name='date'))
    result = df.groupby('group').resample('2D')[['first_val', 'second_val']].mean()
    mi_exp = pd.MultiIndex.from_arrays([[0, 0, 1, 1], df.index[::2]], names=['group', 'date'])
    expected = DataFrame(data={'first_val': [2.0, 2.5, 7.0, 4.0], 'second_val': [4.5, 4.5, 5.0, 4.5]}, index=mi_exp)
    tm.assert_frame_equal(result, expected)