import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_shift_with_multiple_periods_and_fill_value():
    df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': [True, True, False, False, True]})
    shifted_df = df.groupby('b')[['a']].shift([0, 1], fill_value=-1)
    expected_df = DataFrame({'a_0': [1, 2, 3, 4, 5], 'a_1': [-1, 1, -1, 3, 2]})
    tm.assert_frame_equal(shifted_df, expected_df)