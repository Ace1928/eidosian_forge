import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
def test_compare_scalar_interval(self, op, interval_array):
    other = interval_array[0]
    result = op(interval_array, other)
    expected = self.elementwise_comparison(op, interval_array, other)
    tm.assert_numpy_array_equal(result, expected)
    other = Interval(interval_array.left[0], interval_array.right[1])
    result = op(interval_array, other)
    expected = self.elementwise_comparison(op, interval_array, other)
    tm.assert_numpy_array_equal(result, expected)