import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.mark.filterwarnings('ignore:invalid value encountered in:RuntimeWarning')
@pytest.mark.parametrize('opname,op_str', [('add', '+'), ('sub', '-'), ('mul', '*'), ('truediv', '/'), ('pow', '**')])
@pytest.mark.parametrize('left_fix,right_fix', [('_array', '_array2'), ('_array_mixed', '_array_mixed2')])
def test_binary_ops(self, request, opname, op_str, left_fix, right_fix):
    left = request.getfixturevalue(left_fix)
    right = request.getfixturevalue(right_fix)

    def testit(left, right, opname, op_str):
        if opname == 'pow':
            left = np.abs(left)
        op = getattr(operator, opname)
        result = expr.evaluate(op, left, left, use_numexpr=True)
        expected = expr.evaluate(op, left, left, use_numexpr=False)
        tm.assert_numpy_array_equal(result, expected)
        result = expr._can_use_numexpr(op, op_str, right, right, 'evaluate')
        assert not result
    with option_context('compute.use_numexpr', False):
        testit(left, right, opname, op_str)
    expr.set_numexpr_threads(1)
    testit(left, right, opname, op_str)
    expr.set_numexpr_threads()
    testit(left, right, opname, op_str)