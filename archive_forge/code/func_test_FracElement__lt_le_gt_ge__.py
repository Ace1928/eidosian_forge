from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracElement__lt_le_gt_ge__():
    F, x, y = field('x,y', ZZ)
    assert F(1) < 1 / x < 1 / x ** 2 < 1 / x ** 3
    assert F(1) <= 1 / x <= 1 / x ** 2 <= 1 / x ** 3
    assert -7 / x < 1 / x < 3 / x < y / x < 1 / x ** 2
    assert -7 / x <= 1 / x <= 3 / x <= y / x <= 1 / x ** 2
    assert 1 / x ** 3 > 1 / x ** 2 > 1 / x > F(1)
    assert 1 / x ** 3 >= 1 / x ** 2 >= 1 / x >= F(1)
    assert 1 / x ** 2 > y / x > 3 / x > 1 / x > -7 / x
    assert 1 / x ** 2 >= y / x >= 3 / x >= 1 / x >= -7 / x