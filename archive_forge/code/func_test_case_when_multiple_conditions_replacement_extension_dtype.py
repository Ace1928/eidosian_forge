import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_multiple_conditions_replacement_extension_dtype(df):
    """
    Test output when replacement has an extension dtype
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([([True, False, False], 1), (df['a'].gt(1) & df['b'].eq(5), pd_array([1, 2, 3], dtype='Int64'))])
    expected = Series([1, 2, np.nan], dtype='Float64')
    tm.assert_series_equal(result, expected)