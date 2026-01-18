import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_shift_with_multiple_periods_and_freq():
    df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': [True, True, False, False, True]}, index=date_range('1/1/2000', periods=5, freq='h'))
    shifted_df = df.groupby('b')[['a']].shift([0, 1], freq='h')
    expected_df = DataFrame({'a_0': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan], 'a_1': [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=date_range('1/1/2000', periods=6, freq='h'))
    tm.assert_frame_equal(shifted_df, expected_df)