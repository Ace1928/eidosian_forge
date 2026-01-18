from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_call():
    T = Poly(cyclotomic_poly(5, x))
    B = PowerBasis(T)
    assert B(0).col.flat() == [1, 0, 0, 0]
    assert B(1).col.flat() == [0, 1, 0, 0]
    col = DomainMatrix.eye(4, ZZ)[:, 2]
    assert B(col).col == col
    raises(ValueError, lambda: B(-1))