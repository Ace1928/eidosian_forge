import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_microsecond(self):
    per = Period('2020-01-30 15:57:27.576166', freq='us')
    assert per.ordinal == 1580399847576166
    start = per.start_time
    expected = Timestamp('2020-01-30 15:57:27.576166')
    assert start == expected
    assert start._value == per.ordinal * 1000
    per2 = Period('2300-01-01', 'us')
    msg = '2300-01-01'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        per2.start_time
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        per2.end_time