import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_freq_mult(self):
    pidx = period_range(start='2014-01', freq='2M', periods=4)
    expected = PeriodIndex(['2014-01', '2014-03', '2014-05', '2014-07'], freq='2M')
    tm.assert_index_equal(pidx, expected)
    pidx = period_range(start='2014-01-02', end='2014-01-15', freq='3D')
    expected = PeriodIndex(['2014-01-02', '2014-01-05', '2014-01-08', '2014-01-11', '2014-01-14'], freq='3D')
    tm.assert_index_equal(pidx, expected)
    pidx = period_range(end='2014-01-01 17:00', freq='4h', periods=3)
    expected = PeriodIndex(['2014-01-01 09:00', '2014-01-01 13:00', '2014-01-01 17:00'], freq='4h')
    tm.assert_index_equal(pidx, expected)
    msg = 'Frequency must be positive, because it represents span: -1M'
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(['2011-01'], freq='-1M')
    msg = 'Frequency must be positive, because it represents span: 0M'
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(['2011-01'], freq='0M')
    msg = 'Frequency must be positive, because it represents span: 0M'
    with pytest.raises(ValueError, match=msg):
        period_range('2011-01', periods=3, freq='0M')