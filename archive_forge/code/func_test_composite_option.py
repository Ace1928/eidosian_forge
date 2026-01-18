from sympy.polys.constructor import construct_domain
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField
from sympy.core import (Catalan, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
def test_composite_option():
    assert construct_domain({(1,): sin(y)}, composite=False) == (EX, {(1,): EX(sin(y))})
    assert construct_domain({(1,): y}, composite=False) == (EX, {(1,): EX(y)})
    assert construct_domain({(1, 1): 1}, composite=False) == (ZZ, {(1, 1): 1})
    assert construct_domain({(1, 0): y}, composite=False) == (EX, {(1, 0): EX(y)})