from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_pow():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    e = A(to_col([0, 2, 0, 0]), denom=3)
    g = C(to_col([0, 0, 0, 1]), denom=2)
    assert e ** 3 == A(to_col([0, 0, 0, 8]), denom=27)
    assert g ** 2 == C(to_col([0, 3, 0, 0]), denom=4)
    assert e ** 0 == A(to_col([1, 0, 0, 0]))
    assert g ** 0 == A(to_col([1, 0, 0, 0]))
    assert e ** 1 == e
    assert g ** 1 == g