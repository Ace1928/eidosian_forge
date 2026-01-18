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
def test_PolyElement_compose():
    R, x = ring('x', ZZ)
    f = x ** 3 + 4 * x ** 2 + 2 * x + 3
    r = f.compose(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    assert f.compose(x, x) == f
    assert f.compose(x, x ** 2) == x ** 6 + 4 * x ** 4 + 2 * x ** 2 + 3
    raises(CoercionFailed, lambda: f.compose(x, QQ(1, 7)))
    R, x, y, z = ring('x,y,z', ZZ)
    f = x ** 3 + 4 * x ** 2 + 2 * x + 3
    r = f.compose(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    r = f.compose([(x, 0), (y, 0)])
    assert r == 3 and isinstance(r, R.dtype)
    r = (x ** 3 + 4 * x ** 2 + 2 * x * y * z + 3).compose(x, y * z ** 2 - 1)
    q = (y * z ** 2 - 1) ** 3 + 4 * (y * z ** 2 - 1) ** 2 + 2 * (y * z ** 2 - 1) * y * z + 3
    assert r == q and isinstance(r, R.dtype)