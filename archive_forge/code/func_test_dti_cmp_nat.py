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
@pytest.mark.parametrize('dtype', [None, object])
def test_dti_cmp_nat(self, dtype, box_with_array):
    left = DatetimeIndex([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')])
    right = DatetimeIndex([NaT, NaT, Timestamp('2011-01-03')])
    left = tm.box_expected(left, box_with_array)
    right = tm.box_expected(right, box_with_array)
    xbox = get_upcast_box(left, right, True)
    lhs, rhs = (left, right)
    if dtype is object:
        lhs, rhs = (left.astype(object), right.astype(object))
    result = rhs == lhs
    expected = np.array([False, False, True])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(result, expected)
    result = lhs != rhs
    expected = np.array([True, True, False])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(result, expected)
    expected = np.array([False, False, False])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(lhs == NaT, expected)
    tm.assert_equal(NaT == rhs, expected)
    expected = np.array([True, True, True])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(lhs != NaT, expected)
    tm.assert_equal(NaT != lhs, expected)
    expected = np.array([False, False, False])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(lhs < NaT, expected)
    tm.assert_equal(NaT > lhs, expected)