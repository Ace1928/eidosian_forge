from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_find_min_poly():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    powers = []
    m = find_min_poly(A(1), QQ, x=x, powers=powers)
    assert m == Poly(T, domain=QQ)
    assert len(powers) == 5
    m = find_min_poly(A(1), QQ, x=x)
    assert m == Poly(T, domain=QQ)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    raises(MissingUnityError, lambda: find_min_poly(B(1), QQ))