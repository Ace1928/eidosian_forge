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
def test_minpoly_compose():
    eq = S('\n        -1/(800*sqrt(-1/240 + 1/(18000*(-1/17280000 +\n        sqrt(15)*I/28800000)**(1/3)) + 2*(-1/17280000 +\n        sqrt(15)*I/28800000)**(1/3)))')
    mp = minimal_polynomial(eq + 3, x)
    assert mp == 8000 * x ** 2 - 48000 * x + 71999
    assert minimal_polynomial(exp(I * pi / 8), x) == x ** 8 + 1
    mp = minimal_polynomial(sin(pi / 7) + sqrt(2), x)
    assert mp == 4096 * x ** 12 - 63488 * x ** 10 + 351488 * x ** 8 - 826496 * x ** 6 + 770912 * x ** 4 - 268432 * x ** 2 + 28561
    mp = minimal_polynomial(cos(pi / 7) + sqrt(2), x)
    assert mp == 64 * x ** 6 - 64 * x ** 5 - 432 * x ** 4 + 304 * x ** 3 + 712 * x ** 2 - 232 * x - 239
    mp = minimal_polynomial(exp(I * pi / 7) + sqrt(2), x)
    assert mp == x ** 12 - 2 * x ** 11 - 9 * x ** 10 + 16 * x ** 9 + 43 * x ** 8 - 70 * x ** 7 - 97 * x ** 6 + 126 * x ** 5 + 211 * x ** 4 - 212 * x ** 3 - 37 * x ** 2 + 142 * x + 127
    mp = minimal_polynomial(sin(pi / 7) + sqrt(2), x)
    assert mp == 4096 * x ** 12 - 63488 * x ** 10 + 351488 * x ** 8 - 826496 * x ** 6 + 770912 * x ** 4 - 268432 * x ** 2 + 28561
    mp = minimal_polynomial(cos(pi / 7) + sqrt(2), x)
    assert mp == 64 * x ** 6 - 64 * x ** 5 - 432 * x ** 4 + 304 * x ** 3 + 712 * x ** 2 - 232 * x - 239
    mp = minimal_polynomial(exp(I * pi / 7) + sqrt(2), x)
    assert mp == x ** 12 - 2 * x ** 11 - 9 * x ** 10 + 16 * x ** 9 + 43 * x ** 8 - 70 * x ** 7 - 97 * x ** 6 + 126 * x ** 5 + 211 * x ** 4 - 212 * x ** 3 - 37 * x ** 2 + 142 * x + 127
    mp = minimal_polynomial(exp(I * pi * Rational(2, 7)), x)
    assert mp == x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x + 1
    mp = minimal_polynomial(exp(I * pi * Rational(2, 15)), x)
    assert mp == x ** 8 - x ** 7 + x ** 5 - x ** 4 + x ** 3 - x + 1
    mp = minimal_polynomial(cos(pi * Rational(2, 7)), x)
    assert mp == 8 * x ** 3 + 4 * x ** 2 - 4 * x - 1
    mp = minimal_polynomial(sin(pi * Rational(2, 7)), x)
    ex = (5 * cos(pi * Rational(2, 7)) - 7) / (9 * cos(pi / 7) - 5 * cos(pi * Rational(3, 7)))
    mp = minimal_polynomial(ex, x)
    assert mp == x ** 3 + 2 * x ** 2 - x - 1
    assert minimal_polynomial(-1 / (2 * cos(pi / 7)), x) == x ** 3 + 2 * x ** 2 - x - 1
    assert minimal_polynomial(sin(pi * Rational(2, 15)), x) == 256 * x ** 8 - 448 * x ** 6 + 224 * x ** 4 - 32 * x ** 2 + 1
    assert minimal_polynomial(sin(pi * Rational(5, 14)), x) == 8 * x ** 3 - 4 * x ** 2 - 4 * x + 1
    assert minimal_polynomial(cos(pi / 15), x) == 16 * x ** 4 + 8 * x ** 3 - 16 * x ** 2 - 8 * x + 1
    ex = rootof(x ** 3 + x * 4 + 1, 0)
    mp = minimal_polynomial(ex, x)
    assert mp == x ** 3 + 4 * x + 1
    mp = minimal_polynomial(ex + 1, x)
    assert mp == x ** 3 - 3 * x ** 2 + 7 * x - 4
    assert minimal_polynomial(exp(I * pi / 3), x) == x ** 2 - x + 1
    assert minimal_polynomial(exp(I * pi / 4), x) == x ** 4 + 1
    assert minimal_polynomial(exp(I * pi / 6), x) == x ** 4 - x ** 2 + 1
    assert minimal_polynomial(exp(I * pi / 9), x) == x ** 6 - x ** 3 + 1
    assert minimal_polynomial(exp(I * pi / 10), x) == x ** 8 - x ** 6 + x ** 4 - x ** 2 + 1
    assert minimal_polynomial(sin(pi / 9), x) == 64 * x ** 6 - 96 * x ** 4 + 36 * x ** 2 - 3
    assert minimal_polynomial(sin(pi / 11), x) == 1024 * x ** 10 - 2816 * x ** 8 + 2816 * x ** 6 - 1232 * x ** 4 + 220 * x ** 2 - 11
    assert minimal_polynomial(sin(pi / 21), x) == 4096 * x ** 12 - 11264 * x ** 10 + 11264 * x ** 8 - 4992 * x ** 6 + 960 * x ** 4 - 64 * x ** 2 + 1
    assert minimal_polynomial(cos(pi / 9), x) == 8 * x ** 3 - 6 * x - 1
    ex = 2 ** Rational(1, 3) * exp(2 * I * pi / 3)
    assert minimal_polynomial(ex, x) == x ** 3 - 2
    raises(NotAlgebraic, lambda: minimal_polynomial(cos(pi * sqrt(2)), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(sin(pi * sqrt(2)), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(exp(1.618 * I * pi), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(exp(I * pi * sqrt(2)), x))
    ex = 1 / (-36000 - 7200 * sqrt(5) + (12 * sqrt(10) * sqrt(sqrt(5) + 5) + 24 * sqrt(10) * sqrt(-sqrt(5) + 5)) ** 2) + 1
    raises(ZeroDivisionError, lambda: minimal_polynomial(ex, x))
    ex = sqrt(1 + 2 ** Rational(1, 3)) + sqrt(1 + 2 ** Rational(1, 4)) + sqrt(2)
    mp = minimal_polynomial(ex, x)
    assert degree(mp) == 48 and mp.subs({x: 0}) == -16630256576
    ex = tan(pi / 5, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == x ** 4 - 10 * x ** 2 + 5
    assert mp.subs(x, tan(pi / 5)).is_zero
    ex = tan(pi / 6, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == 3 * x ** 2 - 1
    assert mp.subs(x, tan(pi / 6)).is_zero
    ex = tan(pi / 10, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == 5 * x ** 4 - 10 * x ** 2 + 1
    assert mp.subs(x, tan(pi / 10)).is_zero
    raises(NotAlgebraic, lambda: minimal_polynomial(tan(pi * sqrt(2)), x))