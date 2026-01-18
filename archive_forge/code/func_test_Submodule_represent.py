from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Submodule_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    a0 = A(to_col([6, 12, 18, 24]))
    a1 = A(to_col([2, 4, 6, 8]))
    a2 = A(to_col([1, 3, 5, 7]))
    b1 = B.represent(a1)
    assert b1.flat() == [1, 2, 3, 4]
    c0 = C.represent(a0)
    assert c0.flat() == [1, 2, 3, 4]
    Y = A.submodule_from_matrix(DomainMatrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], (3, 4), ZZ).transpose())
    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    z0 = Z(to_col([1, 2, 3, 4, 5, 6]))
    raises(ClosureFailure, lambda: Y.represent(A(3)))
    raises(ClosureFailure, lambda: B.represent(a2))
    raises(ClosureFailure, lambda: B.represent(z0))