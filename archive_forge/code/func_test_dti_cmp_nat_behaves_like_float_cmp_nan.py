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
def test_dti_cmp_nat_behaves_like_float_cmp_nan(self):
    fidx1 = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
    fidx2 = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])
    didx1 = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
    didx2 = DatetimeIndex(['2014-02-01', '2014-03-01', NaT, NaT, '2014-06-01', '2014-07-01'])
    darr = np.array([np.datetime64('2014-02-01 00:00'), np.datetime64('2014-03-01 00:00'), np.datetime64('nat'), np.datetime64('nat'), np.datetime64('2014-06-01 00:00'), np.datetime64('2014-07-01 00:00')])
    cases = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]
    with tm.assert_produces_warning(None):
        for idx1, idx2 in cases:
            result = idx1 < idx2
            expected = np.array([True, False, False, False, True, False])
            tm.assert_numpy_array_equal(result, expected)
            result = idx2 > idx1
            expected = np.array([True, False, False, False, True, False])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 <= idx2
            expected = np.array([True, False, False, False, True, True])
            tm.assert_numpy_array_equal(result, expected)
            result = idx2 >= idx1
            expected = np.array([True, False, False, False, True, True])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 == idx2
            expected = np.array([False, False, False, False, False, True])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 != idx2
            expected = np.array([True, True, True, True, True, False])
            tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(None):
        for idx1, val in [(fidx1, np.nan), (didx1, NaT)]:
            result = idx1 < val
            expected = np.array([False, False, False, False, False, False])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 > val
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 <= val
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 >= val
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 == val
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 != val
            expected = np.array([True, True, True, True, True, True])
            tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(None):
        for idx1, val in [(fidx1, 3), (didx1, datetime(2014, 3, 1))]:
            result = idx1 < val
            expected = np.array([True, False, False, False, False, False])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 > val
            expected = np.array([False, False, False, False, True, True])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 <= val
            expected = np.array([True, False, True, False, False, False])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 >= val
            expected = np.array([False, False, True, False, True, True])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 == val
            expected = np.array([False, False, True, False, False, False])
            tm.assert_numpy_array_equal(result, expected)
            result = idx1 != val
            expected = np.array([True, True, False, True, True, True])
            tm.assert_numpy_array_equal(result, expected)