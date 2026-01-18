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
@pytest.mark.parametrize('offset_kwargs, expected_arg', [({'nanoseconds': 1}, '1970-01-01 00:00:00.000000001'), ({'nanoseconds': 5}, '1970-01-01 00:00:00.000000005'), ({'nanoseconds': -1}, '1969-12-31 23:59:59.999999999'), ({'microseconds': 1}, '1970-01-01 00:00:00.000001'), ({'microseconds': -1}, '1969-12-31 23:59:59.999999'), ({'seconds': 1}, '1970-01-01 00:00:01'), ({'seconds': -1}, '1969-12-31 23:59:59'), ({'minutes': 1}, '1970-01-01 00:01:00'), ({'minutes': -1}, '1969-12-31 23:59:00'), ({'hours': 1}, '1970-01-01 01:00:00'), ({'hours': -1}, '1969-12-31 23:00:00'), ({'days': 1}, '1970-01-02 00:00:00'), ({'days': -1}, '1969-12-31 00:00:00'), ({'weeks': 1}, '1970-01-08 00:00:00'), ({'weeks': -1}, '1969-12-25 00:00:00'), ({'months': 1}, '1970-02-01 00:00:00'), ({'months': -1}, '1969-12-01 00:00:00'), ({'years': 1}, '1971-01-01 00:00:00'), ({'years': -1}, '1969-01-01 00:00:00')])
def test_dateoffset_add_sub(offset_kwargs, expected_arg):
    offset = DateOffset(**offset_kwargs)
    ts = Timestamp(0)
    result = ts + offset
    expected = Timestamp(expected_arg)
    assert result == expected
    result -= offset
    assert result == ts
    result = offset + ts
    assert result == expected