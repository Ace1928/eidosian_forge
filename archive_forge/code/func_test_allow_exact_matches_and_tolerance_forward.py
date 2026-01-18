import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_allow_exact_matches_and_tolerance_forward(self):
    left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    right = pd.DataFrame({'a': [1, 3, 4, 6, 11], 'right_val': [1, 3, 4, 6, 11]})
    expected = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'right_val': [np.nan, 6, 11]})
    result = merge_asof(left, right, on='a', direction='forward', allow_exact_matches=False, tolerance=1)
    tm.assert_frame_equal(result, expected)