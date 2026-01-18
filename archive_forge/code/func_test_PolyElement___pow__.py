from functools import reduce
from operator import add, mul
from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import (oo, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
def test_PolyElement___pow__():
    R, x = ring('x', ZZ, grlex)
    f = 2 * x + 3
    assert f ** 0 == 1
    assert f ** 1 == f
    raises(ValueError, lambda: f ** (-1))
    assert x ** (-1) == x ** (-1)
    assert f ** 2 == f._pow_generic(2) == f._pow_multinomial(2) == 4 * x ** 2 + 12 * x + 9
    assert f ** 3 == f._pow_generic(3) == f._pow_multinomial(3) == 8 * x ** 3 + 36 * x ** 2 + 54 * x + 27
    assert f ** 4 == f._pow_generic(4) == f._pow_multinomial(4) == 16 * x ** 4 + 96 * x ** 3 + 216 * x ** 2 + 216 * x + 81
    assert f ** 5 == f._pow_generic(5) == f._pow_multinomial(5) == 32 * x ** 5 + 240 * x ** 4 + 720 * x ** 3 + 1080 * x ** 2 + 810 * x + 243
    R, x, y, z = ring('x,y,z', ZZ, grlex)
    f = x ** 3 * y - 2 * x * y ** 2 - 3 * z + 1
    g = x ** 6 * y ** 2 - 4 * x ** 4 * y ** 3 - 6 * x ** 3 * y * z + 2 * x ** 3 * y + 4 * x ** 2 * y ** 4 + 12 * x * y ** 2 * z - 4 * x * y ** 2 + 9 * z ** 2 - 6 * z + 1
    assert f ** 2 == f._pow_generic(2) == f._pow_multinomial(2) == g
    R, t = ring('t', ZZ)
    f = -11200 * t ** 4 - 2604 * t ** 2 + 49
    g = 15735193600000000 * t ** 16 + 14633730048000000 * t ** 14 + 4828147466240000 * t ** 12 + 598976863027200 * t ** 10 + 3130812416256 * t ** 8 - 2620523775744 * t ** 6 + 92413760096 * t ** 4 - 1225431984 * t ** 2 + 5764801
    assert f ** 4 == f._pow_generic(4) == f._pow_multinomial(4) == g