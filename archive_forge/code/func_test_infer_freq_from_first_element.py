import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_infer_freq_from_first_element(self):
    msg = 'Period with BDay freq is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        start = Period('02-Apr-2005', 'B')
        end_intv = Period('2005-05-01', 'B')
        period_range(start=start, end=end_intv)
        i2 = PeriodIndex([end_intv, Period('2005-05-05', 'B')])
    assert len(i2) == 2
    assert i2[0] == end_intv
    with tm.assert_produces_warning(FutureWarning, match=msg):
        i2 = PeriodIndex(np.array([end_intv, Period('2005-05-05', 'B')]))
    assert len(i2) == 2
    assert i2[0] == end_intv