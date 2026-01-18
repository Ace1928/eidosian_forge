from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
@pytest.mark.skipif(missing_import, reason='pyneqsys.symbolic req. missing')
def test_SymbolicSys():
    a, b, t = sp.symbols('a b t')

    def f(x):
        return 1 / (x + a) ** t + b
    neqsys = SymbolicSys([a, b], [f(0) - 1, f(1) - 0], [t])
    ab, sol = neqsys.solve([0.5, -0.5], 1, solver='scipy')
    assert sol['success']
    assert abs(ab[0] - (-1 / 2 + 5 ** 0.5 / 2)) < 1e-10
    assert abs(ab[1] - (1 / 2 - 5 ** 0.5 / 2)) < 1e-10