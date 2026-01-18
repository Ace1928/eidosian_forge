from sympy.solvers.decompogen import decompogen, compogen
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import XFAIL, raises
@XFAIL
def test_decompogen_fails():
    A = lambda x: x ** 2 + 2 * x + 3
    B = lambda x: 4 * x ** 2 + 5 * x + 6
    assert decompogen(A(x * exp(x)), x) == [x ** 2 + 2 * x + 3, x * exp(x)]
    assert decompogen(A(B(x)), x) == [x ** 2 + 2 * x + 3, 4 * x ** 2 + 5 * x + 6]
    assert decompogen(A(1 / x + 1 / x ** 2), x) == [x ** 2 + 2 * x + 3, 1 / x + 1 / x ** 2]
    assert decompogen(A(1 / x + 2 / (x + 1)), x) == [x ** 2 + 2 * x + 3, 1 / x + 2 / (x + 1)]