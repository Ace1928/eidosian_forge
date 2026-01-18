import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_simple_new_empty(self):
    idx = PeriodIndex([], freq='M', name='p')
    with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
        idx._simple_new(idx, name='p')
    result = idx._simple_new(idx._data, name='p')
    tm.assert_index_equal(result, idx)