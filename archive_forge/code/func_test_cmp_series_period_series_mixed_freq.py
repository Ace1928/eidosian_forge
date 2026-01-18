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
def test_cmp_series_period_series_mixed_freq(self):
    base = Series([Period('2011', freq='Y'), Period('2011-02', freq='M'), Period('2013', freq='Y'), Period('2011-04', freq='M')])
    ser = Series([Period('2012', freq='Y'), Period('2011-01', freq='M'), Period('2013', freq='Y'), Period('2011-05', freq='M')])
    exp = Series([False, False, True, False])
    tm.assert_series_equal(base == ser, exp)
    exp = Series([True, True, False, True])
    tm.assert_series_equal(base != ser, exp)
    exp = Series([False, True, False, False])
    tm.assert_series_equal(base > ser, exp)
    exp = Series([True, False, False, True])
    tm.assert_series_equal(base < ser, exp)
    exp = Series([False, True, True, False])
    tm.assert_series_equal(base >= ser, exp)
    exp = Series([True, False, True, True])
    tm.assert_series_equal(base <= ser, exp)