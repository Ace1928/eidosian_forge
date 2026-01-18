import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil
def test_compare_with_booleanarray(self, comparison_op, dtype):
    op = comparison_op
    left = pd.array([True, False, None] * 3, dtype='boolean')
    right = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype=dtype)
    other = pd.array([False] * 3 + [True] * 3 + [None] * 3, dtype='boolean')
    expected = op(left, other)
    result = op(left, right)
    tm.assert_extension_array_equal(result, expected)
    expected = op(other, left)
    result = op(right, left)
    tm.assert_extension_array_equal(result, expected)