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
def test_PolyElement_degrees():
    R, x, y, z = ring('x,y,z', ZZ)
    assert R(0).degrees() == (-oo, -oo, -oo)
    assert R(1).degrees() == (0, 0, 0)
    assert (x ** 2 * y + x ** 3 * z ** 2).degrees() == (3, 1, 2)