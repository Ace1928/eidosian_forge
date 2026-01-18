import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_asfreq_near_zero_weekly(self):
    per1 = Period('0001-01-01', 'D') + 6
    per2 = Period('0001-01-01', 'D') - 6
    week1 = per1.asfreq('W')
    week2 = per2.asfreq('W')
    assert week1 != week2
    assert week1.asfreq('D', 'E') >= per1
    assert week2.asfreq('D', 'S') <= per2