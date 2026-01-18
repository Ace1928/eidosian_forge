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
def test_PolyRing_mul():
    R, x = ring('x', ZZ)
    F = [x ** 2 + 2 * i + 3 for i in range(4)]
    assert R.mul(F) == reduce(mul, F) == x ** 8 + 24 * x ** 6 + 206 * x ** 4 + 744 * x ** 2 + 945
    R, = ring('', ZZ)
    assert R.mul([2, 3, 5]) == 30