from datetime import datetime
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
def test_constructor_datetime_nonns(self, constructor):
    arr = np.array(['2020-01-01T00:00:00.000000'], dtype='datetime64[us]')
    dta = pd.core.arrays.DatetimeArray._simple_new(arr, dtype=arr.dtype)
    expected = constructor(dta)
    assert expected.dtype == arr.dtype
    result = constructor(arr)
    tm.assert_equal(result, expected)
    arr.flags.writeable = False
    result = constructor(arr)
    tm.assert_equal(result, expected)