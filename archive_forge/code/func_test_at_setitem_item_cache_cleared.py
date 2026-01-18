from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_setitem_item_cache_cleared(self):
    df = DataFrame(index=[0])
    df['x'] = 1
    df['cost'] = 2
    df['cost']
    df.loc[[0]]
    df.at[0, 'x'] = 4
    df.at[0, 'cost'] = 789
    expected = DataFrame({'x': [4], 'cost': 789}, index=[0], columns=Index(['x', 'cost'], dtype=object))
    tm.assert_frame_equal(df, expected)
    tm.assert_series_equal(df['cost'], expected['cost'])