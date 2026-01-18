from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_interval_mult(self, closed):
    interval = Interval(0, 1, closed=closed)
    expected = Interval(0, 2, closed=closed)
    result = interval * 2
    assert result == expected
    result = 2 * interval
    assert result == expected
    result = interval
    result *= 2
    assert result == expected
    msg = 'unsupported operand type\\(s\\) for \\*'
    with pytest.raises(TypeError, match=msg):
        interval * interval
    msg = "can\\'t multiply sequence by non-int"
    with pytest.raises(TypeError, match=msg):
        interval * 'foo'