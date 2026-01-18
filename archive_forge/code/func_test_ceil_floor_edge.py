from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.parametrize('test_input, rounder, freq, expected', [('2117-01-01 00:00:45', 'floor', '15s', '2117-01-01 00:00:45'), ('2117-01-01 00:00:45', 'ceil', '15s', '2117-01-01 00:00:45'), ('2117-01-01 00:00:45.000000012', 'floor', '10ns', '2117-01-01 00:00:45.000000010'), ('1823-01-01 00:00:01.000000012', 'ceil', '10ns', '1823-01-01 00:00:01.000000020'), ('1823-01-01 00:00:01', 'floor', '1s', '1823-01-01 00:00:01'), ('1823-01-01 00:00:01', 'ceil', '1s', '1823-01-01 00:00:01'), ('NaT', 'floor', '1s', 'NaT'), ('NaT', 'ceil', '1s', 'NaT')])
def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
    dt = Timestamp(test_input)
    func = getattr(dt, rounder)
    result = func(freq)
    if dt is NaT:
        assert result is NaT
    else:
        expected = Timestamp(expected)
        assert result == expected