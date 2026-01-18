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
def test_issue_10198():
    assert reduce_inequalities(-1 + 1 / abs(1 / x - 1) < 0) == (x > -oo) & (x < S(1) / 2) & Ne(x, 0)
    assert reduce_inequalities(abs(1 / sqrt(x)) - 1, x) == Eq(x, 1)
    assert reduce_abs_inequality(-3 + 1 / abs(1 - 1 / x), '<', x) == Or(And(-oo < x, x < 0), And(S.Zero < x, x < Rational(3, 4)), And(Rational(3, 2) < x, x < oo))
    raises(ValueError, lambda: reduce_abs_inequality(-3 + 1 / abs(1 - 1 / sqrt(x)), '<', x))