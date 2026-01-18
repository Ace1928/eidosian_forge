from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_submodule_from_gens():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    gens = [2 * A(0), 2 * A(1), 6 * A(0), 6 * A(1)]
    B = A.submodule_from_gens(gens)
    M = gens[0].column().hstack(gens[1].column())
    assert B.matrix == M
    raises(ValueError, lambda: A.submodule_from_gens([]))
    raises(ValueError, lambda: A.submodule_from_gens([3 * A(0), B(0)]))