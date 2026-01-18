from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)
from sympy.testing.pytest import XFAIL
from sympy.abc import x, y
@XFAIL
def test_issue_6811_fail():
    xp, y, x, z = symbols('xp, y, x, z')
    eq = 4 * (-19 * sin(x) * y + 5 * sin(3 * x) * y + 15 * cos(2 * x) * z - 21 * z) * xp / (9 * cos(x) - 5 * cos(3 * x))
    assert trigsimp(eq) == -2 * (2 * cos(x) * tan(x) * y + 3 * z) * xp / cos(x)