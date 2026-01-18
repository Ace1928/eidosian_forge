from sympy.core import S, Rational
from sympy.integrals.quadrature import (gauss_legendre, gauss_laguerre,
def test_lobatto():
    x, w = gauss_lobatto(2, 17)
    assert [str(r) for r in x] == ['-1', '1']
    assert [str(r) for r in w] == ['1.0000000000000000', '1.0000000000000000']
    x, w = gauss_lobatto(3, 17)
    assert [str(r) for r in x] == ['-1', '0', '1']
    assert [str(r) for r in w] == ['0.33333333333333333', '1.3333333333333333', '0.33333333333333333']
    x, w = gauss_lobatto(4, 17)
    assert [str(r) for r in x] == ['-1', '-0.44721359549995794', '0.44721359549995794', '1']
    assert [str(r) for r in w] == ['0.16666666666666667', '0.83333333333333333', '0.83333333333333333', '0.16666666666666667']
    x, w = gauss_lobatto(5, 17)
    assert [str(r) for r in x] == ['-1', '-0.65465367070797714', '0', '0.65465367070797714', '1']
    assert [str(r) for r in w] == ['0.10000000000000000', '0.54444444444444444', '0.71111111111111111', '0.54444444444444444', '0.10000000000000000']