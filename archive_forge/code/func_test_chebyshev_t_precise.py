from sympy.core import S, Rational
from sympy.integrals.quadrature import (gauss_legendre, gauss_laguerre,
def test_chebyshev_t_precise():
    x, w = gauss_chebyshev_t(3, 40)
    assert [str(r) for r in x] == ['0.8660254037844386467637231707529361834714', '0', '-0.8660254037844386467637231707529361834714']
    assert [str(r) for r in w] == ['1.047197551196597746154214461093167628066', '1.047197551196597746154214461093167628066', '1.047197551196597746154214461093167628066']