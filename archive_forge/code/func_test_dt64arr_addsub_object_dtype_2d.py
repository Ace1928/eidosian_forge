from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dt64arr_addsub_object_dtype_2d():
    dti = date_range('1994-02-13', freq='2W', periods=4)
    dta = dti._data.reshape((4, 1))
    other = np.array([[pd.offsets.Day(n)] for n in range(4)])
    assert other.shape == dta.shape
    with tm.assert_produces_warning(PerformanceWarning):
        result = dta + other
    with tm.assert_produces_warning(PerformanceWarning):
        expected = (dta[:, 0] + other[:, 0]).reshape(-1, 1)
    tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(PerformanceWarning):
        result2 = dta - dta.astype(object)
    assert result2.shape == (4, 1)
    assert all((td._value == 0 for td in result2.ravel()))