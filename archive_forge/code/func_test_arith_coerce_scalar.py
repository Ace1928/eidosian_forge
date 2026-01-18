import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_arith_coerce_scalar(data, all_arithmetic_operators):
    op = tm.get_op_from_name(all_arithmetic_operators)
    s = pd.Series(data)
    other = 0.01
    result = op(s, other)
    expected = op(s.astype(float), other)
    expected = expected.astype('Float64')
    if all_arithmetic_operators == '__rmod__':
        mask = (s == 0).fillna(False).to_numpy(bool)
        expected.array._mask[mask] = False
    tm.assert_series_equal(result, expected)