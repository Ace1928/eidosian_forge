from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_comparison_tuples(self):
    s = Series([(1, 1), (1, 2)])
    result = s == (1, 2)
    expected = Series([False, True])
    tm.assert_series_equal(result, expected)
    result = s != (1, 2)
    expected = Series([True, False])
    tm.assert_series_equal(result, expected)
    result = s == (0, 0)
    expected = Series([False, False])
    tm.assert_series_equal(result, expected)
    result = s != (0, 0)
    expected = Series([True, True])
    tm.assert_series_equal(result, expected)
    s = Series([(1, 1), (1, 1)])
    result = s == (1, 1)
    expected = Series([True, True])
    tm.assert_series_equal(result, expected)
    result = s != (1, 1)
    expected = Series([False, False])
    tm.assert_series_equal(result, expected)