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
def test_PolyRing_add():
    R, x = ring('x', ZZ)
    F = [x ** 2 + 2 * i + 3 for i in range(4)]
    assert R.add(F) == reduce(add, F) == 4 * x ** 2 + 24
    R, = ring('', ZZ)
    assert R.add([2, 5, 7]) == 14