import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_incompat_freq(self):
    msg = 'Input has different freq=D from PeriodIndex\\(freq=M\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        PeriodIndex([Period('2011-01', freq='M'), NaT, Period('2011-01', freq='D')])
    with pytest.raises(IncompatibleFrequency, match=msg):
        PeriodIndex(np.array([Period('2011-01', freq='M'), NaT, Period('2011-01', freq='D')]))
    with pytest.raises(IncompatibleFrequency, match=msg):
        PeriodIndex([NaT, Period('2011-01', freq='M'), Period('2011-01', freq='D')])
    with pytest.raises(IncompatibleFrequency, match=msg):
        PeriodIndex(np.array([NaT, Period('2011-01', freq='M'), Period('2011-01', freq='D')]))