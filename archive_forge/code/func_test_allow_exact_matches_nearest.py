import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_allow_exact_matches_nearest(self):
    left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    right = pd.DataFrame({'a': [1, 2, 3, 7, 11], 'right_val': [1, 2, 3, 7, 11]})
    expected = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'right_val': [2, 3, 11]})
    result = merge_asof(left, right, on='a', direction='nearest', allow_exact_matches=False)
    tm.assert_frame_equal(result, expected)