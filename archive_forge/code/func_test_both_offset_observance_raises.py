from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_both_offset_observance_raises():
    msg = 'Cannot use both offset and observance'
    with pytest.raises(NotImplementedError, match=msg):
        Holiday('Cyber Monday', month=11, day=1, offset=[DateOffset(weekday=SA(4))], observance=next_monday)