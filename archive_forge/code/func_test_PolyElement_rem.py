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
def test_PolyElement_rem():
    R, x = ring('x', ZZ, grlex)
    f = x ** 3 - 12 * x ** 2 - 42
    g = x - 3
    r = -123
    assert f.rem([g]) == f.div([g])[1] == r
    R, x, y = ring('x,y', ZZ, grlex)
    f = 4 * x ** 2 * y - 2 * x * y + 4 * x - 2 * y + 8
    assert f.rem([R(2)]) == f.div([R(2)])[1] == 0
    assert f.rem([2 * y]) == f.div([2 * y])[1] == 4 * x + 8
    f = x - 1
    g = y - 1
    assert f.rem([g]) == f.div([g])[1] == f
    f = x * y ** 2 + 1
    G = [x * y + 1, y + 1]
    r = 2
    assert f.rem(G) == f.div(G)[1] == r
    f = x ** 2 * y + x * y ** 2 + y ** 2
    G = [x * y - 1, y ** 2 - 1]
    r = x + y + 1
    assert f.rem(G) == f.div(G)[1] == r
    G = [y ** 2 - 1, x * y - 1]
    r = 2 * x + 1
    assert f.rem(G) == f.div(G)[1] == r