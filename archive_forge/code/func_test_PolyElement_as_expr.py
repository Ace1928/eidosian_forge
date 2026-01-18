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
def test_PolyElement_as_expr():
    R, x, y, z = ring('x,y,z', ZZ)
    f = 3 * x ** 2 * y - x * y * z + 7 * z ** 3 + 1
    X, Y, Z = R.symbols
    g = 3 * X ** 2 * Y - X * Y * Z + 7 * Z ** 3 + 1
    assert f != g
    assert f.as_expr() == g
    U, V, W = symbols('u,v,w')
    g = 3 * U ** 2 * V - U * V * W + 7 * W ** 3 + 1
    assert f != g
    assert f.as_expr(U, V, W) == g
    raises(ValueError, lambda: f.as_expr(X))
    R, = ring('', ZZ)
    assert R(3).as_expr() == 3