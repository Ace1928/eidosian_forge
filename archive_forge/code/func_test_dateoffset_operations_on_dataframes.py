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
def test_dateoffset_operations_on_dataframes():
    df = DataFrame({'T': [Timestamp('2019-04-30')], 'D': [DateOffset(months=1)]})
    frameresult1 = df['T'] + 26 * df['D']
    df2 = DataFrame({'T': [Timestamp('2019-04-30'), Timestamp('2019-04-30')], 'D': [DateOffset(months=1), DateOffset(months=1)]})
    expecteddate = Timestamp('2021-06-30')
    with tm.assert_produces_warning(PerformanceWarning):
        frameresult2 = df2['T'] + 26 * df2['D']
    assert frameresult1[0] == expecteddate
    assert frameresult2[0] == expecteddate