from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement___mul__():
    F, x, y = field('x,y', QQ)
    f, g = (1 / x, 1 / y)
    assert f * g == g * f == 1 / (x * y)
    assert x * F.ring.gens[0] == F.ring.gens[0] * x == x ** 2
    F, x, y = field('x,y', ZZ)
    assert x * 3 == 3 * x
    assert x * QQ(3, 7) == QQ(3, 7) * x == x * Rational(3, 7)
    Fuv, u, v = field('u,v', ZZ)
    Fxyzt, x, y, z, t = field('x,y,z,t', Fuv)
    f = ((u + 1) * x * y + 1) / ((v - 1) * z - t * u * v - 1)
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u * v, (0, 0, 0, 0): -1}
    Ruv, u, v = ring('u,v', ZZ)
    Fxyzt, x, y, z, t = field('x,y,z,t', Ruv)
    f = ((u + 1) * x * y + 1) / ((v - 1) * z - t * u * v - 1)
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u * v, (0, 0, 0, 0): -1}