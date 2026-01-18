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
def test_PolyElement_symmetrize():
    R, x, y = ring('x,y', ZZ)
    f = x ** 2 + y ** 2
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f
    f = x ** 2 - y ** 2
    sym, rem, m = f.symmetrize()
    assert rem != 0
    assert sym.compose(m) + rem == f
    f = x * y + 7
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f
    f = y + 7
    sym, rem, m = f.symmetrize()
    assert rem != 0
    assert sym.compose(m) + rem == f
    f = R.from_expr(3)
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f
    R, f = sring(3)
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f