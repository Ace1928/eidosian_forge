from sympy.core.function import expand
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (AlgebraicNumber, I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solveset import nonlinsolve
from sympy.geometry import Circle, intersection
from sympy.testing.pytest import raises, slow
from sympy.sets.sets import FiniteSet
from sympy.geometry.point import Point2D
from sympy.polys.numberfields.minpoly import (
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import (
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import rootof
from sympy.polys.polytools import degree
from sympy.abc import x, y, z
def test_issue_19760():
    e = 1 / (sqrt(1 + sqrt(2)) - sqrt(2) * sqrt(1 + sqrt(2))) + 1
    mp_expected = x ** 4 - 4 * x ** 3 + 4 * x ** 2 - 2
    for comp in (True, False):
        mp = Poly(minimal_polynomial(e, compose=comp))
        assert mp(x) == mp_expected, 'minimal_polynomial(e, compose=%s) = %s; %s expected' % (comp, mp(x), mp_expected)