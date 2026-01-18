from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_mod():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(to_col([1, 15, 8, 0]), denom=2)
    assert e % 7 == A(to_col([1, 1, 8, 0]), denom=2)
    assert e % QQ(1, 2) == A.zero()
    assert e % QQ(1, 3) == A(to_col([1, 1, 0, 0]), denom=6)
    B = A.submodule_from_gens([A(0), 5 * A(1), 3 * A(2), A(3)])
    assert e % B == A(to_col([1, 5, 2, 0]), denom=2)
    C = B.whole_submodule()
    raises(TypeError, lambda: e % C)