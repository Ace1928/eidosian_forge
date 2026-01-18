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
def test_PolyElement_discriminant():
    _, x = ring('x', ZZ)
    f, g = (x ** 3 + 3 * x ** 2 + 9 * x - 13, -11664)
    assert f.discriminant() == g
    F, a, b, c = ring('a,b,c', ZZ)
    _, x = ring('x', F)
    f, g = (a * x ** 2 + b * x + c, b ** 2 - 4 * a * c)
    assert f.discriminant() == g