from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
@pytest.mark.parametrize('ts', [Timestamp('1776-07-04'), Timestamp('1776-07-04', tz='UTC')])
@pytest.mark.parametrize('other', [1, np.int64(1), np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.uint64)])
def test_add_int_with_freq(self, ts, other):
    msg = 'Addition/subtraction of integers and integer-arrays'
    with pytest.raises(TypeError, match=msg):
        ts + other
    with pytest.raises(TypeError, match=msg):
        other + ts
    with pytest.raises(TypeError, match=msg):
        ts - other
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        other - ts