from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import delta_to_tick
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_Nanosecond():
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert_offset_equal(Nano(), timestamp, timestamp + np.timedelta64(1, 'ns'))
    assert_offset_equal(Nano(-1), timestamp + np.timedelta64(1, 'ns'), timestamp)
    assert_offset_equal(2 * Nano(), timestamp, timestamp + np.timedelta64(2, 'ns'))
    assert_offset_equal(-1 * Nano(), timestamp + np.timedelta64(1, 'ns'), timestamp)
    assert Nano(3) + Nano(2) == Nano(5)
    assert Nano(3) - Nano(2) == Nano()
    assert Nano(1) + Nano(10) == Nano(11)
    assert Nano(5) + Micro(1) == Nano(1005)
    assert Micro(5) + Nano(1) == Nano(5001)