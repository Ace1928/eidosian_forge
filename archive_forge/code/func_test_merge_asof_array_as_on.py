import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_asof_array_as_on(unit):
    dti = pd.DatetimeIndex(['2021/01/01 00:37', '2021/01/01 01:40'], dtype=f'M8[{unit}]')
    right = pd.DataFrame({'a': [2, 6], 'ts': dti})
    ts_merge = pd.date_range(start=pd.Timestamp('2021/01/01 00:00'), periods=3, freq='1h', unit=unit)
    left = pd.DataFrame({'b': [4, 8, 7]})
    result = merge_asof(left, right, left_on=ts_merge, right_on='ts', allow_exact_matches=False, direction='backward')
    expected = pd.DataFrame({'b': [4, 8, 7], 'a': [np.nan, 2, 6], 'ts': ts_merge})
    tm.assert_frame_equal(result, expected)
    result = merge_asof(right, left, left_on='ts', right_on=ts_merge, allow_exact_matches=False, direction='backward')
    expected = pd.DataFrame({'a': [2, 6], 'ts': dti, 'b': [4, 8]})
    tm.assert_frame_equal(result, expected)