import pytest
from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_no_rounding_occurs(self, tz_naive_fixture):
    tz = tz_naive_fixture
    rng = date_range(start='2016-01-01', periods=5, freq='2Min', tz=tz)
    expected_rng = DatetimeIndex([Timestamp('2016-01-01 00:00:00', tz=tz), Timestamp('2016-01-01 00:02:00', tz=tz), Timestamp('2016-01-01 00:04:00', tz=tz), Timestamp('2016-01-01 00:06:00', tz=tz), Timestamp('2016-01-01 00:08:00', tz=tz)]).as_unit('ns')
    result = rng.round(freq='2min')
    tm.assert_index_equal(result, expected_rng)