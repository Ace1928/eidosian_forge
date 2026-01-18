import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_asfreq_corner(self):
    val = Period(freq='Y', year=2007)
    result1 = val.asfreq('5min')
    result2 = val.asfreq('min')
    expected = Period('2007-12-31 23:59', freq='min')
    assert result1.ordinal == expected.ordinal
    assert result1.freqstr == '5min'
    assert result2.ordinal == expected.ordinal
    assert result2.freqstr == 'min'