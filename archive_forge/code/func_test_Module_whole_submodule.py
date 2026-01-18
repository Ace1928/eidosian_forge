from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_whole_submodule():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.whole_submodule()
    e = B(to_col([1, 2, 3, 4]))
    f = e.to_parent()
    assert f.col.flat() == [1, 2, 3, 4]
    e0, e1, e2, e3 = (B(0), B(1), B(2), B(3))
    assert e2 * e3 == e0
    assert e3 ** 2 == e1