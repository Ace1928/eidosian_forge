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
def test_PolyElement_gcdex():
    _, x = ring('x', QQ)
    f, g = (2 * x, x ** 2 - 16)
    s, t, h = (x / 32, -QQ(1, 16), 1)
    assert f.half_gcdex(g) == (s, h)
    assert f.gcdex(g) == (s, t, h)