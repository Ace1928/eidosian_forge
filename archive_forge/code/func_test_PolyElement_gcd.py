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
def test_PolyElement_gcd():
    R, x, y = ring('x,y', QQ)
    f = QQ(1, 2) * x ** 2 + x + QQ(1, 2)
    g = QQ(1, 2) * x + QQ(1, 2)
    assert f.gcd(g) == x + 1