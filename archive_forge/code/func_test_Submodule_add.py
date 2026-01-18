from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Submodule_add():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(DomainMatrix([[4, 0, 0, 0], [0, 4, 0, 0]], (2, 4), ZZ).transpose(), denom=6)
    C = A.submodule_from_matrix(DomainMatrix([[0, 10, 0, 0], [0, 0, 7, 0]], (2, 4), ZZ).transpose(), denom=15)
    D = A.submodule_from_matrix(DomainMatrix([[20, 0, 0, 0], [0, 20, 0, 0], [0, 0, 14, 0]], (3, 4), ZZ).transpose(), denom=30)
    assert B + C == D
    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    Y = Z.submodule_from_gens([Z(0), Z(1)])
    raises(TypeError, lambda: B + Y)