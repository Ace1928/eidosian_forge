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
def test_PolyElement_factor_list():
    _, x = ring('x', ZZ)
    f = x ** 5 - x ** 3 - x ** 2 + 1
    u = x + 1
    v = x - 1
    w = x ** 2 + x + 1
    assert f.factor_list() == (1, [(u, 1), (v, 2), (w, 1)])