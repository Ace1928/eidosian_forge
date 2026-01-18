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
def test_PolyElement_evaluate():
    R, x = ring('x', ZZ)
    f = x ** 3 + 4 * x ** 2 + 2 * x + 3
    r = f.evaluate(x, 0)
    assert r == 3 and (not isinstance(r, PolyElement))
    raises(CoercionFailed, lambda: f.evaluate(x, QQ(1, 7)))
    R, x, y, z = ring('x,y,z', ZZ)
    f = (x * y) ** 3 + 4 * (x * y) ** 2 + 2 * x * y + 3
    r = f.evaluate(x, 0)
    assert r == 3 and isinstance(r, R.drop(x).dtype)
    r = f.evaluate([(x, 0), (y, 0)])
    assert r == 3 and isinstance(r, R.drop(x, y).dtype)
    r = f.evaluate(y, 0)
    assert r == 3 and isinstance(r, R.drop(y).dtype)
    r = f.evaluate([(y, 0), (x, 0)])
    assert r == 3 and isinstance(r, R.drop(y, x).dtype)
    r = f.evaluate([(x, 0), (y, 0), (z, 0)])
    assert r == 3 and (not isinstance(r, PolyElement))
    raises(CoercionFailed, lambda: f.evaluate([(x, 1), (y, QQ(1, 7))]))
    raises(CoercionFailed, lambda: f.evaluate([(x, QQ(1, 7)), (y, 1)]))
    raises(CoercionFailed, lambda: f.evaluate([(x, QQ(1, 7)), (y, QQ(1, 7))]))