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
def test_minpoly_groebner():
    assert _minpoly_groebner(S(2) / 3, x, Poly) == 3 * x - 2
    assert _minpoly_groebner((sqrt(2) + 3) * (sqrt(2) + 1), x, Poly) == x ** 2 - 10 * x - 7
    assert _minpoly_groebner((sqrt(2) + 3) ** (S(1) / 3) * (sqrt(2) + 1) ** (S(1) / 3), x, Poly) == x ** 6 - 10 * x ** 3 - 7
    assert _minpoly_groebner((sqrt(2) + 3) ** (-S(1) / 3) * (sqrt(2) + 1) ** (S(1) / 3), x, Poly) == 7 * x ** 6 - 2 * x ** 3 - 1
    raises(NotAlgebraic, lambda: _minpoly_groebner(pi ** 2, x, Poly))