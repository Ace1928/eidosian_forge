from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_PowerBasis_represent():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    col = to_col([1, 2, 3, 4])
    a = A(col)
    assert A.represent(a) == col
    b = A(col, denom=2)
    raises(ClosureFailure, lambda: A.represent(b))