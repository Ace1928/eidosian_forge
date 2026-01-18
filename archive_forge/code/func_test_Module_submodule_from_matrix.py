from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_submodule_from_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    e = B(to_col([1, 2, 3, 4]))
    f = e.to_parent()
    assert f.col.flat() == [2, 4, 6, 8]
    raises(ValueError, lambda: A.submodule_from_matrix(DomainMatrix.eye(4, QQ)))
    raises(ValueError, lambda: A.submodule_from_matrix(2 * DomainMatrix.eye(5, ZZ)))