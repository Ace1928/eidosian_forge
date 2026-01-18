from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_compatibility():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    assert C(0).is_compat(C(1)) is True
    assert C(0).is_compat(D(0)) is False
    u, v = C(0).unify(D(0))
    assert u.module is B and v.module is B
    assert C(C.represent(u)) == C(0) and D(D.represent(v)) == D(0)
    u, v = C(0).unify(C(1))
    assert u == C(0) and v == C(1)
    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(UnificationFailed, lambda: C(0).unify(Z(1)))