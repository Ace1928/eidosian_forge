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
def test_PolyElement_gff_list():
    _, x = ring('x', ZZ)
    f = x ** 5 + 2 * x ** 4 - x ** 3 - 2 * x ** 2
    assert f.gff_list() == [(x, 1), (x + 2, 4)]
    f = x * (x - 1) ** 3 * (x - 2) ** 2 * (x - 4) ** 2 * (x - 5)
    assert f.gff_list() == [(x ** 2 - 5 * x + 4, 1), (x ** 2 - 5 * x + 4, 2), (x, 3)]