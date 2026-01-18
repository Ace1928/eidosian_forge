import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_freq_combined(self):
    for freq in ['1D1h', '1h1D']:
        pidx = PeriodIndex(['2016-01-01', '2016-01-02'], freq=freq)
        expected = PeriodIndex(['2016-01-01 00:00', '2016-01-02 00:00'], freq='25h')
    for freq in ['1D1h', '1h1D']:
        pidx = period_range(start='2016-01-01', periods=2, freq=freq)
        expected = PeriodIndex(['2016-01-01 00:00', '2016-01-02 01:00'], freq='25h')
        tm.assert_index_equal(pidx, expected)