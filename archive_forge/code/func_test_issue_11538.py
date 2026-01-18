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
def test_issue_11538():
    for n in [E, pi, Catalan]:
        assert construct_domain(n)[0] == ZZ[n]
        assert construct_domain(x + n)[0] == ZZ[x, n]
    assert construct_domain(GoldenRatio)[0] == EX
    assert construct_domain(x + GoldenRatio)[0] == EX