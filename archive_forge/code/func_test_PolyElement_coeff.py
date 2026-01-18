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
def test_PolyElement_coeff():
    R, x, y, z = ring('x,y,z', ZZ, lex)
    f = 3 * x ** 2 * y - x * y * z + 7 * z ** 3 + 23
    assert f.coeff(1) == 23
    raises(ValueError, lambda: f.coeff(3))
    assert f.coeff(x) == 0
    assert f.coeff(y) == 0
    assert f.coeff(z) == 0
    assert f.coeff(x ** 2 * y) == 3
    assert f.coeff(x * y * z) == -1
    assert f.coeff(z ** 3) == 7
    raises(ValueError, lambda: f.coeff(3 * x ** 2 * y))
    raises(ValueError, lambda: f.coeff(-x * y * z))
    raises(ValueError, lambda: f.coeff(7 * z ** 3))
    R, = ring('', ZZ)
    assert R(3).coeff(1) == 3