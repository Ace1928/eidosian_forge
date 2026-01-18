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
def test_PolyElement_sqf_norm():
    R, x = ring('x', QQ.algebraic_field(sqrt(3)))
    X = R.to_ground().x
    assert (x ** 2 - 2).sqf_norm() == (1, x ** 2 - 2 * sqrt(3) * x + 1, X ** 4 - 10 * X ** 2 + 1)
    R, x = ring('x', QQ.algebraic_field(sqrt(2)))
    X = R.to_ground().x
    assert (x ** 2 - 3).sqf_norm() == (1, x ** 2 - 2 * sqrt(2) * x - 1, X ** 4 - 10 * X ** 2 + 1)