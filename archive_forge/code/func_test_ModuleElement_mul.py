from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_mul():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    f = A(to_col([0, 0, 0, 7]), denom=5)
    g = C(to_col([0, 0, 0, 1]), denom=2)
    h = A(to_col([0, 0, 3, 1]), denom=7)
    assert e * f == A(to_col([-14, -14, -14, -14]), denom=15)
    assert e * g == A(to_col([-1, -1, -1, -1]))
    assert e * h == A(to_col([-2, -2, -2, 4]), denom=21)
    assert e * QQ(6, 5) == A(to_col([0, 4, 0, 0]), denom=5)
    assert (g * QQ(10, 21)).equiv(A(to_col([0, 0, 0, 5]), denom=7))
    assert e // QQ(6, 5) == A(to_col([0, 5, 0, 0]), denom=9)
    U = Poly(cyclotomic_poly(7, x))
    Z = PowerBasis(U)
    raises(TypeError, lambda: e * Z(0))
    raises(TypeError, lambda: e * 3.14)
    raises(TypeError, lambda: e // 3.14)
    raises(ZeroDivisionError, lambda: e // 0)