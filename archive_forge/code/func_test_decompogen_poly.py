from sympy.solvers.decompogen import decompogen, compogen
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import XFAIL, raises
def test_decompogen_poly():
    assert decompogen(x ** 4 + 2 * x ** 2 + 1, x) == [x ** 2 + 2 * x + 1, x ** 2]
    assert decompogen(x ** 4 + 2 * x ** 3 - x - 1, x) == [x ** 2 - x - 1, x ** 2 + x]