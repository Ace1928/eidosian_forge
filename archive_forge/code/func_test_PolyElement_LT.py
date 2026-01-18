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
def test_PolyElement_LT():
    R, x, y = ring('x,y', QQ, lex)
    assert R(0).LT == ((0, 0), QQ(0))
    assert (QQ(1, 2) * x).LT == ((1, 0), QQ(1, 2))
    assert (QQ(1, 4) * x * y + QQ(1, 2) * x).LT == ((1, 1), QQ(1, 4))
    R, = ring('', ZZ)
    assert R(0).LT == ((), 0)
    assert R(1).LT == ((), 1)