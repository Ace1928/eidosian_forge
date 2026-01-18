from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_addition_subtraction_types(self):
    dt = datetime(2014, 3, 4)
    td = timedelta(seconds=1)
    ts = Timestamp(dt)
    msg = 'Addition/subtraction of integers'
    with pytest.raises(TypeError, match=msg):
        ts + 1
    with pytest.raises(TypeError, match=msg):
        ts - 1
    assert type(ts - dt) == Timedelta
    assert type(ts + td) == Timestamp
    assert type(ts - td) == Timestamp
    td64 = np.timedelta64(1, 'D')
    assert type(ts + td64) == Timestamp
    assert type(ts - td64) == Timestamp