import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_drops_levels_for_one_row_dataframe():
    mi = MultiIndex.from_arrays([['x'], ['y'], ['z']], names=['a', 'b', 'c'])
    df = DataFrame({'d': [0]}, index=mi)
    expected = df.droplevel([0, 2])
    result = df.loc['x', :, 'z']
    tm.assert_frame_equal(result, expected)
    ser = Series([0], index=mi)
    result = ser.loc['x', :, 'z']
    expected = Series([0], index=Index(['y'], name='b'))
    tm.assert_series_equal(result, expected)