from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.and_, operator.or_, operator.xor])
def test_logical_ops_with_index(self, op):
    ser = Series([True, True, False, False])
    idx1 = Index([True, False, True, False])
    idx2 = Index([1, 0, 1, 0])
    expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])
    result = op(ser, idx1)
    tm.assert_series_equal(result, expected)
    expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)
    result = op(ser, idx2)
    tm.assert_series_equal(result, expected)