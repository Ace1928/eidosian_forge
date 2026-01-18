import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.mark.parametrize('dtype', ['Int64', pytest.param('int64[pyarrow]', marks=td.skip_if_no('pyarrow')), pytest.param('timestamp[s][pyarrow]', marks=td.skip_if_no('pyarrow'))])
def test_merge_asof_extension_dtype(dtype):
    left = pd.DataFrame({'join_col': [1, 3, 5], 'left_val': [1, 2, 3]})
    right = pd.DataFrame({'join_col': [2, 3, 4], 'right_val': [1, 2, 3]})
    left = left.astype({'join_col': dtype})
    right = right.astype({'join_col': dtype})
    result = merge_asof(left, right, on='join_col')
    expected = pd.DataFrame({'join_col': [1, 3, 5], 'left_val': [1, 2, 3], 'right_val': [np.nan, 2.0, 3.0]})
    expected = expected.astype({'join_col': dtype})
    tm.assert_frame_equal(result, expected)