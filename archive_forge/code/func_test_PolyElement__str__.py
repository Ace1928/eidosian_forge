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
def test_PolyElement__str__():
    x, y = symbols('x, y')
    for dom in [ZZ, QQ, ZZ[x], ZZ[x, y], ZZ[x][y]]:
        R, t = ring('t', dom)
        assert str(2 * t ** 2 + 1) == '2*t**2 + 1'
    for dom in [EX, EX[x]]:
        R, t = ring('t', dom)
        assert str(2 * t ** 2 + 1) == 'EX(2)*t**2 + EX(1)'