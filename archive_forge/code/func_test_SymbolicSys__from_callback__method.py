from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
@pytest.mark.skipif(missing_import, reason='pyneqsys.symbolic req. missing')
def test_SymbolicSys__from_callback__method():

    class Problem(object):

        def f(self, x, p):
            return [x[0] ** p[0]]
    p = Problem()
    ss = SymbolicSys.from_callback(p.f, 1, 1)
    x, sol = ss.solve([1], [3])
    assert abs(x[0]) < 1e-14