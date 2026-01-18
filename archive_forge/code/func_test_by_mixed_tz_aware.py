import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_by_mixed_tz_aware(self, using_infer_string):
    left = pd.DataFrame({'by_col1': pd.DatetimeIndex(['2018-01-01']).tz_localize('UTC'), 'by_col2': ['HELLO'], 'on_col': [2], 'value': ['a']})
    right = pd.DataFrame({'by_col1': pd.DatetimeIndex(['2018-01-01']).tz_localize('UTC'), 'by_col2': ['WORLD'], 'on_col': [1], 'value': ['b']})
    result = merge_asof(left, right, by=['by_col1', 'by_col2'], on='on_col')
    expected = pd.DataFrame([[pd.Timestamp('2018-01-01', tz='UTC'), 'HELLO', 2, 'a']], columns=['by_col1', 'by_col2', 'on_col', 'value_x'])
    expected['value_y'] = np.array([np.nan], dtype=object)
    if using_infer_string:
        expected['value_y'] = expected['value_y'].astype('string[pyarrow_numpy]')
    tm.assert_frame_equal(result, expected)