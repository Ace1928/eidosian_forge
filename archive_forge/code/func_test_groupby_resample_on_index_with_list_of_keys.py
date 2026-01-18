from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_groupby_resample_on_index_with_list_of_keys():
    df = DataFrame(data={'group': [0, 0, 0, 0, 1, 1, 1, 1], 'val': [3, 1, 4, 1, 5, 9, 2, 6]}, index=date_range(start='2016-01-01', periods=8, name='date'))
    result = df.groupby('group').resample('2D')[['val']].mean()
    mi_exp = pd.MultiIndex.from_arrays([[0, 0, 1, 1], df.index[::2]], names=['group', 'date'])
    expected = DataFrame(data={'val': [2.0, 2.5, 7.0, 4.0]}, index=mi_exp)
    tm.assert_frame_equal(result, expected)