from sympy.core.function import (Function, Lambda, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.solvers.recurr import rsolve, rsolve_hyper, rsolve_poly, rsolve_ratio
from sympy.testing.pytest import raises, slow, XFAIL
from sympy.abc import a, b
@slow
def test_issue_15751():
    f = y(n) + 21 * y(n + 1) - 273 * y(n + 2) - 1092 * y(n + 3) + 1820 * y(n + 4) + 1092 * y(n + 5) - 273 * y(n + 6) - 21 * y(n + 7) + y(n + 8)
    assert rsolve(f, y(n)) is not None