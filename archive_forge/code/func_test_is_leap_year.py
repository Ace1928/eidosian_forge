from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['Y', 'M', 'D', 'h'])
def test_is_leap_year(self, freq):
    p = Period('2000-01-01 00:00:00', freq=freq)
    assert p.is_leap_year
    assert isinstance(p.is_leap_year, bool)
    p = Period('1999-01-01 00:00:00', freq=freq)
    assert not p.is_leap_year
    p = Period('2004-01-01 00:00:00', freq=freq)
    assert p.is_leap_year
    p = Period('2100-01-01 00:00:00', freq=freq)
    assert not p.is_leap_year