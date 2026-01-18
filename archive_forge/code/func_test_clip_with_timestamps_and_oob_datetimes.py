from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [object, 'M8[us]'])
def test_clip_with_timestamps_and_oob_datetimes(self, dtype):
    ser = Series([datetime(1, 1, 1), datetime(9999, 9, 9)], dtype=dtype)
    result = ser.clip(lower=Timestamp.min, upper=Timestamp.max)
    expected = Series([Timestamp.min, Timestamp.max], dtype=dtype)
    tm.assert_series_equal(result, expected)