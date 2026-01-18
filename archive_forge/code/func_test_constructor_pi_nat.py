import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_pi_nat(self):
    idx = PeriodIndex([Period('2011-01', freq='M'), NaT, Period('2011-01', freq='M')])
    exp = PeriodIndex(['2011-01', 'NaT', '2011-01'], freq='M')
    tm.assert_index_equal(idx, exp)
    idx = PeriodIndex(np.array([Period('2011-01', freq='M'), NaT, Period('2011-01', freq='M')]))
    tm.assert_index_equal(idx, exp)
    idx = PeriodIndex([NaT, NaT, Period('2011-01', freq='M'), Period('2011-01', freq='M')])
    exp = PeriodIndex(['NaT', 'NaT', '2011-01', '2011-01'], freq='M')
    tm.assert_index_equal(idx, exp)
    idx = PeriodIndex(np.array([NaT, NaT, Period('2011-01', freq='M'), Period('2011-01', freq='M')]))
    tm.assert_index_equal(idx, exp)
    idx = PeriodIndex([NaT, NaT, '2011-01', '2011-01'], freq='M')
    tm.assert_index_equal(idx, exp)
    with pytest.raises(ValueError, match='freq not specified'):
        PeriodIndex([NaT, NaT])
    with pytest.raises(ValueError, match='freq not specified'):
        PeriodIndex(np.array([NaT, NaT]))
    with pytest.raises(ValueError, match='freq not specified'):
        PeriodIndex(['NaT', 'NaT'])
    with pytest.raises(ValueError, match='freq not specified'):
        PeriodIndex(np.array(['NaT', 'NaT']))