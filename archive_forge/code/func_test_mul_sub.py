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
@pytest.mark.parametrize('arithmatic_offset_type, n, expected', zip(_ARITHMETIC_DATE_OFFSET, range(1, 10), ['2007-01-02', '2007-11-02', '2007-12-12', '2007-12-29', '2008-01-01 19:00:00', '2008-01-01 23:54:00', '2008-01-01 23:59:53', '2008-01-01 23:59:59.992000000', '2008-01-01 23:59:59.999991000']))
def test_mul_sub(self, arithmatic_offset_type, n, expected, dt):
    assert dt - DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
    assert dt - n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)