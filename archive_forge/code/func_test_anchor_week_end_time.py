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
def test_anchor_week_end_time(self):

    def _ex(*args):
        return Timestamp(Timestamp(datetime(*args)).as_unit('ns')._value - 1)
    p = Period('2013-1-1', 'W-SAT')
    xp = _ex(2013, 1, 6)
    assert p.end_time == xp