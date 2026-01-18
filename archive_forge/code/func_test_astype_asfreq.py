import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_astype_asfreq(self):
    pi1 = PeriodIndex(['2011-01-01', '2011-02-01', '2011-03-01'], freq='D')
    exp = PeriodIndex(['2011-01', '2011-02', '2011-03'], freq='M')
    tm.assert_index_equal(pi1.asfreq('M'), exp)
    tm.assert_index_equal(pi1.astype('period[M]'), exp)
    exp = PeriodIndex(['2011-01', '2011-02', '2011-03'], freq='3M')
    tm.assert_index_equal(pi1.asfreq('3M'), exp)
    tm.assert_index_equal(pi1.astype('period[3M]'), exp)