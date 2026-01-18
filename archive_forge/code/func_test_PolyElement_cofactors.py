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
def test_PolyElement_cofactors():
    R, x, y = ring('x,y', ZZ)
    f, g = (R(0), R(0))
    assert f.cofactors(g) == (0, 0, 0)
    f, g = (R(2), R(0))
    assert f.cofactors(g) == (2, 1, 0)
    f, g = (R(-2), R(0))
    assert f.cofactors(g) == (2, -1, 0)
    f, g = (R(0), R(-2))
    assert f.cofactors(g) == (2, 0, -1)
    f, g = (R(0), 2 * x + 4)
    assert f.cofactors(g) == (2 * x + 4, 0, 1)
    f, g = (2 * x + 4, R(0))
    assert f.cofactors(g) == (2 * x + 4, 1, 0)
    f, g = (R(2), R(2))
    assert f.cofactors(g) == (2, 1, 1)
    f, g = (R(-2), R(2))
    assert f.cofactors(g) == (2, -1, 1)
    f, g = (R(2), R(-2))
    assert f.cofactors(g) == (2, 1, -1)
    f, g = (R(-2), R(-2))
    assert f.cofactors(g) == (2, -1, -1)
    f, g = (x ** 2 + 2 * x + 1, R(1))
    assert f.cofactors(g) == (1, x ** 2 + 2 * x + 1, 1)
    f, g = (x ** 2 + 2 * x + 1, R(2))
    assert f.cofactors(g) == (1, x ** 2 + 2 * x + 1, 2)
    f, g = (2 * x ** 2 + 4 * x + 2, R(2))
    assert f.cofactors(g) == (2, x ** 2 + 2 * x + 1, 1)
    f, g = (R(2), 2 * x ** 2 + 4 * x + 2)
    assert f.cofactors(g) == (2, 1, x ** 2 + 2 * x + 1)
    f, g = (2 * x ** 2 + 4 * x + 2, x + 1)
    assert f.cofactors(g) == (x + 1, 2 * x + 2, 1)
    f, g = (x + 1, 2 * x ** 2 + 4 * x + 2)
    assert f.cofactors(g) == (x + 1, 1, 2 * x + 2)
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    f, g = (t ** 2 + 2 * t + 1, 2 * t + 2)
    assert f.cofactors(g) == (t + 1, t + 1, 2)
    f, g = (z ** 2 * t ** 2 + 2 * z ** 2 * t + z ** 2 + z * t + z, t ** 2 + 2 * t + 1)
    h, cff, cfg = (t + 1, z ** 2 * t + z ** 2 + z, t + 1)
    assert f.cofactors(g) == (h, cff, cfg)
    assert g.cofactors(f) == (h, cfg, cff)
    R, x, y = ring('x,y', QQ)
    f = QQ(1, 2) * x ** 2 + x + QQ(1, 2)
    g = QQ(1, 2) * x + QQ(1, 2)
    h = x + 1
    assert f.cofactors(g) == (h, g, QQ(1, 2))
    assert g.cofactors(f) == (h, QQ(1, 2), g)
    R, x, y = ring('x,y', RR)
    f = 2.1 * x * y ** 2 - 2.1 * x * y + 2.1 * x
    g = 2.1 * x ** 3
    h = 1.0 * x
    assert f.cofactors(g) == (h, f / h, g / h)
    assert g.cofactors(f) == (h, g / h, f / h)