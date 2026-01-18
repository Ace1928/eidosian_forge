import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('opname, exp', [('add', [1, 3, None, None, 9]), ('mul', [0, 2, None, None, 20])], ids=['add', 'mul'])
def test_add_mul(dtype, opname, exp):
    a = pd.array([0, 1, None, 3, 4], dtype=dtype)
    b = pd.array([1, 2, 3, None, 5], dtype=dtype)
    expected = pd.array(exp, dtype=dtype)
    op = getattr(operator, opname)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)
    op = getattr(ops, 'r' + opname)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)