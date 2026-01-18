import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_datetime64arr(self):
    vals = np.arange(100000, 100000 + 10000, 100, dtype=np.int64)
    vals = vals.view(np.dtype('M8[us]'))
    pi = PeriodIndex(vals, freq='D')
    expected = PeriodIndex(vals.astype('M8[ns]'), freq='D')
    tm.assert_index_equal(pi, expected)