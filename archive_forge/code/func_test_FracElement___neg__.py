from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement___neg__():
    F, x, y = field('x,y', QQ)
    f = (7 * x - 9) / y
    g = (-7 * x + 9) / y
    assert -f == g
    assert -g == f