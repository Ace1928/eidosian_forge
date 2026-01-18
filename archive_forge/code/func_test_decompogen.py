from sympy.solvers.decompogen import decompogen, compogen
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import XFAIL, raises
def test_decompogen():
    assert decompogen(sin(cos(x)), x) == [sin(x), cos(x)]
    assert decompogen(sin(x) ** 2 + sin(x) + 1, x) == [x ** 2 + x + 1, sin(x)]
    assert decompogen(sqrt(6 * x ** 2 - 5), x) == [sqrt(x), 6 * x ** 2 - 5]
    assert decompogen(sin(sqrt(cos(x ** 2 + 1))), x) == [sin(x), sqrt(x), cos(x), x ** 2 + 1]
    assert decompogen(Abs(cos(x) ** 2 + 3 * cos(x) - 4), x) == [Abs(x), x ** 2 + 3 * x - 4, cos(x)]
    assert decompogen(sin(x) ** 2 + sin(x) - sqrt(3) / 2, x) == [x ** 2 + x - sqrt(3) / 2, sin(x)]
    assert decompogen(Abs(cos(y) ** 2 + 3 * cos(x) - 4), x) == [Abs(x), 3 * x + cos(y) ** 2 - 4, cos(x)]
    assert decompogen(x, y) == [x]
    assert decompogen(1, x) == [1]
    assert decompogen(Max(3, x), x) == [Max(3, x)]
    raises(TypeError, lambda: decompogen(x < 5, x))
    u = 2 * x + 3
    assert decompogen(Max(sqrt(u), u ** 2), x) == [Max(sqrt(x), x ** 2), u]
    assert decompogen(Max(u, u ** 2, y), x) == [Max(x, x ** 2, y), u]
    assert decompogen(Max(sin(x), u), x) == [Max(2 * x + 3, sin(x))]