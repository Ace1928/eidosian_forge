import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_multiple_conditions(df):
    """
    Test output when booleans are derived from a computation
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([(df.a.eq(1), 1), (Series([False, True, False]), 2)])
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)