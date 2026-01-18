from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_to_ancestors():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    eD = D(0)
    eC = eD.to_parent()
    eB = eD.to_ancestor(B)
    eA = eD.over_power_basis()
    assert eC.module is C and eC.coeffs == [5, 0, 0, 0]
    assert eB.module is B and eB.coeffs == [15, 0, 0, 0]
    assert eA.module is A and eA.coeffs == [30, 0, 0, 0]
    a = A(0)
    raises(ValueError, lambda: a.to_parent())