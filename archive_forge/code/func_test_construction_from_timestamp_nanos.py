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
def test_construction_from_timestamp_nanos(self):
    ts = Timestamp('2022-04-20 09:23:24.123456789')
    per = Period(ts, freq='ns')
    rt = per.to_timestamp()
    assert rt == ts
    dt64 = ts.asm8
    per2 = Period(dt64, freq='ns')
    rt2 = per2.to_timestamp()
    assert rt2.asm8 == dt64