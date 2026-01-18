import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_shift_lose_timezone():
    now_dt = Timestamp.utcnow()
    df = DataFrame({'a': [1, 1], 'date': now_dt})
    result = df.groupby('a').shift(0).iloc[0]
    expected = Series({'date': now_dt}, name=result.name)
    tm.assert_series_equal(result, expected)