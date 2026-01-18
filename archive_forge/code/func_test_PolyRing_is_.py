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
def test_PolyRing_is_():
    R = PolyRing('x', QQ, lex)
    assert R.is_univariate is True
    assert R.is_multivariate is False
    R = PolyRing('x,y,z', QQ, lex)
    assert R.is_univariate is False
    assert R.is_multivariate is True
    R = PolyRing('', QQ, lex)
    assert R.is_univariate is False
    assert R.is_multivariate is False