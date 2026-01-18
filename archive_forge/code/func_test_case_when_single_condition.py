import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_single_condition(df):
    """
    Test output on a single condition.
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([(df.a.eq(1), 1)])
    expected = Series([1, np.nan, np.nan])
    tm.assert_series_equal(result, expected)