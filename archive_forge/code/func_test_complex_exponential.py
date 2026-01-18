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
def test_complex_exponential():
    w = exp(-I * 2 * pi / 3, evaluate=False)
    alg = QQ.algebraic_field(w)
    assert construct_domain([w ** 2, w, 1], extension=True) == (alg, [alg.convert(w ** 2), alg.convert(w), alg.convert(1)])