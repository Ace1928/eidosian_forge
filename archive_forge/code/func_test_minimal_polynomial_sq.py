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
def test_minimal_polynomial_sq():
    from sympy.core.add import Add
    from sympy.core.function import expand_multinomial
    p = expand_multinomial((1 + 5 * sqrt(2) + 2 * sqrt(3)) ** 3)
    mp = minimal_polynomial(p ** Rational(1, 3), x)
    assert mp == x ** 4 - 4 * x ** 3 - 118 * x ** 2 + 244 * x + 1321
    p = expand_multinomial((1 + sqrt(2) - 2 * sqrt(3) + sqrt(7)) ** 3)
    mp = minimal_polynomial(p ** Rational(1, 3), x)
    assert mp == x ** 8 - 8 * x ** 7 - 56 * x ** 6 + 448 * x ** 5 + 480 * x ** 4 - 5056 * x ** 3 + 1984 * x ** 2 + 7424 * x - 3008
    p = Add(*[sqrt(i) for i in range(1, 12)])
    mp = minimal_polynomial(p, x)
    assert mp.subs({x: 0}) == -71965773323122507776