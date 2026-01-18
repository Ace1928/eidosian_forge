import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_compare_zerodim(self, box_with_array):
    pi = period_range('2000', periods=4)
    other = np.array(pi.to_numpy()[0])
    pi = tm.box_expected(pi, box_with_array)
    xbox = get_upcast_box(pi, other, True)
    result = pi <= other
    expected = np.array([True, False, False, False])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(result, expected)