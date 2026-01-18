import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.tests.arrays.masked_shared import (
def test_compare_with_integerarray(self, comparison_op):
    op = comparison_op
    a = pd.array([0, 1, None] * 3, dtype='Int64')
    b = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype='Float64')
    other = b.astype('Int64')
    expected = op(a, other)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)
    expected = op(other, a)
    result = op(b, a)
    tm.assert_extension_array_equal(result, expected)