from sympy.concrete.summations import Sum
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.sets.sets import (FiniteSet, Interval, Union)
from sympy.solvers.inequalities import (reduce_inequalities,
from sympy.polys.rootoftools import rootof
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import solveset
from sympy.abc import x, y
from sympy.core.mod import Mod
from sympy.testing.pytest import raises, XFAIL
def test_issue_5526():
    assert reduce_inequalities(0 <= x + Integral(y ** 2, (y, 1, 3)) - 1, [x]) == (x >= -Integral(y ** 2, (y, 1, 3)) + 1)
    f = Function('f')
    e = Sum(f(x), (x, 1, 3))
    assert reduce_inequalities(0 <= x + e + y ** 2, [x]) == (x >= -y ** 2 - Sum(f(x), (x, 1, 3)))