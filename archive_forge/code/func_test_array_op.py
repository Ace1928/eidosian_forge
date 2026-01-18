import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('opname, exp', [('add', [1.1, 2.2, None, None, 5.5]), ('mul', [0.1, 0.4, None, None, 2.5]), ('sub', [0.9, 1.8, None, None, 4.5]), ('truediv', [10.0, 10.0, None, None, 10.0]), ('floordiv', [9.0, 9.0, None, None, 10.0]), ('mod', [0.1, 0.2, None, None, 0.0])], ids=['add', 'mul', 'sub', 'div', 'floordiv', 'mod'])
def test_array_op(dtype, opname, exp):
    a = pd.array([1.0, 2.0, None, 4.0, 5.0], dtype=dtype)
    b = pd.array([0.1, 0.2, 0.3, None, 0.5], dtype=dtype)
    op = getattr(operator, opname)
    result = op(a, b)
    expected = pd.array(exp, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)