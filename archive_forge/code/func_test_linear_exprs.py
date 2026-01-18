from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
@pytest.mark.skipif(missing_import, reason='pyneqsys.symbolic req. missing')
def test_linear_exprs():
    a, b, c = x = sp.symarray('x', 3)
    coeffs = [[1, 3, -2], [3, 5, 6], [2, 4, 3]]
    vals = [5, 7, 8]
    exprs = linear_exprs(coeffs, x, vals)
    known = [1 * a + 3 * b - 2 * c - 5, 3 * a + 5 * b + 6 * c - 7, 2 * a + 4 * b + 3 * c - 8]
    assert all([(rt - kn).simplify() == 0 for rt, kn in zip(exprs, known)])
    rexprs = linear_exprs(coeffs, x, vals, rref=True)
    rknown = [a + 15, b - 8, c - 2]
    assert all([(rt - kn).simplify() == 0 for rt, kn in zip(rexprs, rknown)])