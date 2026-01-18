import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_int_type_tolerance(self, any_int_dtype):
    left = pd.DataFrame({'a': [0, 10, 20], 'left_val': [1, 2, 3]})
    right = pd.DataFrame({'a': [5, 15, 25], 'right_val': [1, 2, 3]})
    left['a'] = left['a'].astype(any_int_dtype)
    right['a'] = right['a'].astype(any_int_dtype)
    expected = pd.DataFrame({'a': [0, 10, 20], 'left_val': [1, 2, 3], 'right_val': [np.nan, 1.0, 2.0]})
    expected['a'] = expected['a'].astype(any_int_dtype)
    result = merge_asof(left, right, on='a', tolerance=10)
    tm.assert_frame_equal(result, expected)