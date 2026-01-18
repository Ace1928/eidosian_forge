import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_from_ordinals(self):
    Period(ordinal=-1000, freq='Y')
    Period(ordinal=0, freq='Y')
    msg = "The 'ordinal' keyword in PeriodIndex is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx1 = PeriodIndex(ordinal=[-1, 0, 1], freq='Y')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx2 = PeriodIndex(ordinal=np.array([-1, 0, 1]), freq='Y')
    tm.assert_index_equal(idx1, idx2)
    alt1 = PeriodIndex.from_ordinals([-1, 0, 1], freq='Y')
    tm.assert_index_equal(alt1, idx1)
    alt2 = PeriodIndex.from_ordinals(np.array([-1, 0, 1]), freq='Y')
    tm.assert_index_equal(alt2, idx2)