from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_get_offset():
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset('gibberish')
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset('QS-JAN-B')
    pairs = [('B', BDay()), ('b', BDay()), ('bme', BMonthEnd()), ('Bme', BMonthEnd()), ('W-MON', Week(weekday=0)), ('W-TUE', Week(weekday=1)), ('W-WED', Week(weekday=2)), ('W-THU', Week(weekday=3)), ('W-FRI', Week(weekday=4))]
    for name, expected in pairs:
        offset = _get_offset(name)
        assert offset == expected, f'Expected {repr(name)} to yield {repr(expected)} (actual: {repr(offset)})'