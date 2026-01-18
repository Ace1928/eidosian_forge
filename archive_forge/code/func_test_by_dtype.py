import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.mark.parametrize('dtype', ['float64', 'int16', 'm8[ns]', 'M8[us]'])
def test_by_dtype(self, dtype):
    left = pd.DataFrame({'by_col': np.array([1], dtype=dtype), 'on_col': [2], 'value': ['a']})
    right = pd.DataFrame({'by_col': np.array([1], dtype=dtype), 'on_col': [1], 'value': ['b']})
    result = merge_asof(left, right, by='by_col', on='on_col')
    expected = pd.DataFrame({'by_col': np.array([1], dtype=dtype), 'on_col': [2], 'value_x': ['a'], 'value_y': ['b']})
    tm.assert_frame_equal(result, expected)