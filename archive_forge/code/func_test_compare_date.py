from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'US/Pacific'])
def test_compare_date(self, tz):
    ts = Timestamp('2021-01-01 00:00:00.00000', tz=tz)
    dt = ts.to_pydatetime().date()
    msg = 'Cannot compare Timestamp with datetime.date'
    for left, right in [(ts, dt), (dt, ts)]:
        assert not left == right
        assert left != right
        with pytest.raises(TypeError, match=msg):
            left < right
        with pytest.raises(TypeError, match=msg):
            left <= right
        with pytest.raises(TypeError, match=msg):
            left > right
        with pytest.raises(TypeError, match=msg):
            left >= right