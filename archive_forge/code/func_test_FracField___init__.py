from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracField___init__():
    F1 = FracField('x,y', ZZ, lex)
    F2 = FracField('x,y', ZZ, lex)
    F3 = FracField('x,y,z', ZZ, lex)
    assert F1.x == F1.gens[0]
    assert F1.y == F1.gens[1]
    assert F1.x == F2.x
    assert F1.y == F2.y
    assert F1.x != F3.x
    assert F1.y != F3.y