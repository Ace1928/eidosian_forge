from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_equiv():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 2, 3, 4]), denom=1)
    f = A(to_col([3, 6, 9, 12]), denom=3)
    assert e.equiv(f)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    g = C(to_col([1, 2, 3, 4]), denom=1)
    h = A(to_col([3, 6, 9, 12]), denom=1)
    assert g.equiv(h)
    assert C(to_col([5, 0, 0, 0]), denom=7).equiv(QQ(15, 7))
    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(UnificationFailed, lambda: e.equiv(Z(0)))
    assert e.equiv(3.14) is False