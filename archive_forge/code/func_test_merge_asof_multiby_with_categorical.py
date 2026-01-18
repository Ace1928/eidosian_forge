import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_multiby_with_categorical():
    left = pd.DataFrame({'c1': pd.Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b']), 'c2': ['x'] * 4, 't': [1] * 4, 'v': range(4)})
    right = pd.DataFrame({'c1': pd.Categorical(['b', 'b'], categories=['b', 'a']), 'c2': ['x'] * 2, 't': [1, 2], 'v': range(2)})
    result = merge_asof(left, right, by=['c1', 'c2'], on='t', direction='forward', suffixes=['_left', '_right'])
    expected = pd.DataFrame({'c1': pd.Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b']), 'c2': ['x'] * 4, 't': [1] * 4, 'v_left': range(4), 'v_right': [np.nan, np.nan, 0.0, 0.0]})
    tm.assert_frame_equal(result, expected)