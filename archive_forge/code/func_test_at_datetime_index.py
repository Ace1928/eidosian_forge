from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('row', (Timestamp('2019-01-01'), '2019-01-01'))
def test_at_datetime_index(self, row):
    df = DataFrame(data=[[1] * 2], index=DatetimeIndex(data=['2019-01-01', '2019-01-02'])).astype({0: 'float64'})
    expected = DataFrame(data=[[0.5, 1], [1.0, 1]], index=DatetimeIndex(data=['2019-01-01', '2019-01-02']))
    df.at[row, 0] = 0.5
    tm.assert_frame_equal(df, expected)