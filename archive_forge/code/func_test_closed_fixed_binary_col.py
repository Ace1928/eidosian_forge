from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_closed_fixed_binary_col(center, step):
    data = [0, 1, 1, 0, 0, 1, 0, 1]
    df = DataFrame({'binary_col': data}, index=date_range(start='2020-01-01', freq='min', periods=len(data)))
    if center:
        expected_data = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]
    expected = DataFrame(expected_data, columns=['binary_col'], index=date_range(start='2020-01-01', freq='min', periods=len(expected_data)))[::step]
    rolling = df.rolling(window=len(df), closed='left', min_periods=1, center=center, step=step)
    result = rolling.mean()
    tm.assert_frame_equal(result, expected)