import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_base_constructor_with_period_dtype(self):
    dtype = PeriodDtype('D')
    values = ['2011-01-01', '2012-03-04', '2014-05-01']
    result = Index(values, dtype=dtype)
    expected = PeriodIndex(values, dtype=dtype)
    tm.assert_index_equal(result, expected)