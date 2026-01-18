from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_int(self):
    df1 = DataFrame({'a': [0, 1, 3, 5]}, dtype='int64')
    df2 = DataFrame({'a': [1, 4]}, dtype='int64')
    result_12 = df1.combine_first(df2)
    expected_12 = DataFrame({'a': [0, 1, 3, 5]})
    tm.assert_frame_equal(result_12, expected_12)
    result_21 = df2.combine_first(df1)
    expected_21 = DataFrame({'a': [1, 4, 3, 5]})
    tm.assert_frame_equal(result_21, expected_21)