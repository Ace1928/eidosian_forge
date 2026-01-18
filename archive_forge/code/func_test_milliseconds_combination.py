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
@pytest.mark.parametrize('offset_kwargs, expected_arg', [({'microseconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:00.001001'), ({'seconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:01.001'), ({'minutes': 1, 'milliseconds': 1}, '2022-01-01 00:01:00.001'), ({'hours': 1, 'milliseconds': 1}, '2022-01-01 01:00:00.001'), ({'days': 1, 'milliseconds': 1}, '2022-01-02 00:00:00.001'), ({'weeks': 1, 'milliseconds': 1}, '2022-01-08 00:00:00.001'), ({'months': 1, 'milliseconds': 1}, '2022-02-01 00:00:00.001'), ({'years': 1, 'milliseconds': 1}, '2023-01-01 00:00:00.001')])
def test_milliseconds_combination(self, offset_kwargs, expected_arg):
    offset = DateOffset(**offset_kwargs)
    ts = Timestamp('2022-01-01')
    result = ts + offset
    expected = Timestamp(expected_arg)
    assert result == expected