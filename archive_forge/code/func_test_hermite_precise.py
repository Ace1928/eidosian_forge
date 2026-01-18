from sympy.core import S, Rational
from sympy.integrals.quadrature import (gauss_legendre, gauss_laguerre,
def test_hermite_precise():
    x, w = gauss_hermite(3, 40)
    assert [str(r) for r in x] == ['-1.224744871391589049098642037352945695983', '0', '1.224744871391589049098642037352945695983']
    assert [str(r) for r in w] == ['0.2954089751509193378830279138901908637996', '1.181635900603677351532111655560763455198', '0.2954089751509193378830279138901908637996']