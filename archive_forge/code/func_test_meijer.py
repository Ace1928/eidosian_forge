from sympy.core.containers import Tuple
from sympy.core.function import Derivative
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)
from sympy.series.order import O
from sympy.abc import x, z, k
from sympy.series.limits import limit
from sympy.testing.pytest import raises, slow
from sympy.core.random import (
def test_meijer():
    raises(TypeError, lambda: meijerg(1, z))
    raises(TypeError, lambda: meijerg(((1,), (2,)), (3,), (4,), z))
    assert meijerg(((1, 2), (3,)), ((4,), (5,)), z) == meijerg(Tuple(1, 2), Tuple(3), Tuple(4), Tuple(5), z)
    g = meijerg((1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14), z)
    assert g.an == Tuple(1, 2)
    assert g.ap == Tuple(1, 2, 3, 4, 5)
    assert g.aother == Tuple(3, 4, 5)
    assert g.bm == Tuple(6, 7, 8, 9)
    assert g.bq == Tuple(6, 7, 8, 9, 10, 11, 12, 13, 14)
    assert g.bother == Tuple(10, 11, 12, 13, 14)
    assert g.argument == z
    assert g.nu == 75
    assert g.delta == -1
    assert g.is_commutative is True
    assert g.is_number is False
    assert meijerg([[], []], [[S.Half], [0]], 1).is_number is True
    assert meijerg([1, 2], [3], [4], [5], z).delta == S.Half
    assert tn(meijerg(Tuple(), Tuple(), Tuple(0), Tuple(), -z), exp(z), z)
    assert tn(sqrt(pi) * meijerg(Tuple(), Tuple(), Tuple(0), Tuple(S.Half), z ** 2 / 4), cos(z), z)
    assert tn(meijerg(Tuple(1, 1), Tuple(), Tuple(1), Tuple(0), z), log(1 + z), z)
    raises(ValueError, lambda: meijerg(((3, 1), (2,)), ((oo,), (2, 0)), x))
    raises(ValueError, lambda: meijerg(((3, 1), (2,)), ((1,), (2, 0)), x))
    g = meijerg((randcplx(),), (randcplx() + 2 * I,), Tuple(), (randcplx(), randcplx()), z)
    assert td(g, z)
    g = meijerg(Tuple(), (randcplx(),), Tuple(), (randcplx(), randcplx()), z)
    assert td(g, z)
    g = meijerg(Tuple(), Tuple(), Tuple(randcplx()), Tuple(randcplx(), randcplx()), z)
    assert td(g, z)
    a1, a2, b1, b2, c1, c2, d1, d2 = symbols('a1:3, b1:3, c1:3, d1:3')
    assert meijerg((a1, a2), (b1, b2), (c1, c2), (d1, d2), z).diff(z) == (meijerg((a1 - 1, a2), (b1, b2), (c1, c2), (d1, d2), z) + (a1 - 1) * meijerg((a1, a2), (b1, b2), (c1, c2), (d1, d2), z)) / z
    assert meijerg([z, z], [], [], [], z).diff(z) == Derivative(meijerg([z, z], [], [], [], z), z)
    from sympy.functions.elementary.complexes import polar_lift as pl
    assert meijerg([pl(a1)], [pl(a2)], [pl(b1)], [pl(b2)], pl(z)) == meijerg([a1], [a2], [b1], [b2], pl(z))
    from sympy.abc import a, b, c, d, s
    assert meijerg([a], [b], [c], [d], z).integrand(s) == z ** s * gamma(c - s) * gamma(-a + s + 1) / (gamma(b - s) * gamma(-d + s + 1))