from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_return_obj_type_with_bools(self):
    df1 = DataFrame([[np.nan, 3.0, True], [-4.6, np.nan, True], [np.nan, 7.0, False]])
    df2 = DataFrame([[-42.6, np.nan, True], [-5.0, 1.6, False]], index=[1, 2])
    expected = Series([True, True, False], name=2, dtype=bool)
    result_12 = df1.combine_first(df2)[2]
    tm.assert_series_equal(result_12, expected)
    result_21 = df2.combine_first(df1)[2]
    tm.assert_series_equal(result_21, expected)