import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_read_only_ndarray():
    left = pd.Series([2], index=[2], name='left')
    right = pd.Series([1], index=[1], name='right')
    left.index.values.flags.writeable = False
    right.index.values.flags.writeable = False
    result = merge_asof(left, right, left_index=True, right_index=True)
    expected = pd.DataFrame({'left': [2], 'right': [1]}, index=[2])
    tm.assert_frame_equal(result, expected)