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
def test_PolyElement_coeffs():
    R, x, y, z = ring('x,y,z', QQ)
    coeffs = (x ** 2 / 3 + y ** 3 / 4 + z ** 4 / 5).coeffs()
    assert coeffs == [QQ(1, 3), QQ(1, 4), QQ(1, 5)]
    R, x, y = ring('x,y', ZZ, lex)
    f = x * y ** 7 + 2 * x ** 2 * y ** 3
    assert f.coeffs() == f.coeffs(lex) == f.coeffs('lex') == [2, 1]
    assert f.coeffs(grlex) == f.coeffs('grlex') == [1, 2]
    R, x, y = ring('x,y', ZZ, grlex)
    f = x * y ** 7 + 2 * x ** 2 * y ** 3
    assert f.coeffs() == f.coeffs(grlex) == f.coeffs('grlex') == [1, 2]
    assert f.coeffs(lex) == f.coeffs('lex') == [2, 1]