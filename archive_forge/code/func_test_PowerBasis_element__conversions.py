from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_PowerBasis_element__conversions():
    k = QQ.cyclotomic_field(5)
    L = QQ.cyclotomic_field(7)
    B = PowerBasis(k)
    a = k([QQ(1, 2), QQ(1, 3), 5, 7])
    e = B.element_from_ANP(a)
    assert e.coeffs == [42, 30, 2, 3]
    assert e.denom == 6
    assert e.to_ANP() == a
    d = L([QQ(1, 2), QQ(1, 3), 5, 7])
    raises(UnificationFailed, lambda: B.element_from_ANP(d))
    alpha = k.to_alg_num(a)
    eps = B.element_from_alg_num(alpha)
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6
    assert eps.to_alg_num() == alpha
    delta = L.to_alg_num(d)
    raises(UnificationFailed, lambda: B.element_from_alg_num(delta))
    C = PowerBasis(k.ext.minpoly)
    eps = C.element_from_alg_num(alpha)
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6
    raises(StructureError, lambda: eps.to_alg_num())