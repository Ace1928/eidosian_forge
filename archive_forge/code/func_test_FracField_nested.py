from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracField_nested():
    a, b, x = symbols('a b x')
    F1 = ZZ.frac_field(a, b)
    F2 = F1.frac_field(x)
    frac = F2(a + b)
    assert frac.numer == F1.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F1(a + b)]
    assert frac.denom == F1.poly_ring(x)(1)
    F3 = ZZ.poly_ring(a, b)
    F4 = F3.frac_field(x)
    frac = F4(a + b)
    assert frac.numer == F3.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F3(a + b)]
    assert frac.denom == F3.poly_ring(x)(1)
    frac = F2(F3(a + b))
    assert frac.numer == F1.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F1(a + b)]
    assert frac.denom == F1.poly_ring(x)(1)
    frac = F4(F1(a + b))
    assert frac.numer == F3.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F3(a + b)]
    assert frac.denom == F3.poly_ring(x)(1)