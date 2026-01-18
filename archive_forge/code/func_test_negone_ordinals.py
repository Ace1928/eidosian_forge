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
def test_negone_ordinals():
    freqs = ['Y', 'M', 'Q', 'D', 'h', 'min', 's']
    period = Period(ordinal=-1, freq='D')
    for freq in freqs:
        repr(period.asfreq(freq))
    for freq in freqs:
        period = Period(ordinal=-1, freq=freq)
        repr(period)
        assert period.year == 1969
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        period = Period(ordinal=-1, freq='B')
    repr(period)
    period = Period(ordinal=-1, freq='W')
    repr(period)