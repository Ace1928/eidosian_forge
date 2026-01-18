import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_timedelta_quantile():
    df = DataFrame({'value': pd.to_timedelta(np.arange(4), unit='s'), 'group': [1, 1, 2, 2]})
    result = df.groupby('group').quantile(0.99)
    expected = DataFrame({'value': [pd.Timedelta('0 days 00:00:00.990000'), pd.Timedelta('0 days 00:00:02.990000')]}, index=Index([1, 2], name='group'))
    tm.assert_frame_equal(result, expected)