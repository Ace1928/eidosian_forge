import numpy as np
import pytest
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_astype_period():
    arr = period_array(['2000', '2001', None], freq='D')
    result = arr.astype(PeriodDtype('M'))
    expected = period_array(['2000', '2001', None], freq='M')
    tm.assert_period_array_equal(result, expected)