import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_nearest_by(self):
    left = pd.DataFrame({'a': [1, 5, 10, 12, 15], 'b': ['X', 'X', 'Z', 'Z', 'Y'], 'left_val': ['a', 'b', 'c', 'd', 'e']})
    right = pd.DataFrame({'a': [1, 6, 11, 15, 16], 'b': ['X', 'Z', 'Z', 'Z', 'Y'], 'right_val': [1, 6, 11, 15, 16]})
    expected = pd.DataFrame({'a': [1, 5, 10, 12, 15], 'b': ['X', 'X', 'Z', 'Z', 'Y'], 'left_val': ['a', 'b', 'c', 'd', 'e'], 'right_val': [1, 1, 11, 11, 16]})
    result = merge_asof(left, right, on='a', by='b', direction='nearest')
    tm.assert_frame_equal(result, expected)