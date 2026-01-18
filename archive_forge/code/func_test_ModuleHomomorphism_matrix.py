from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleHomomorphism_matrix():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    phi = ModuleEndomorphism(A, lambda a: a ** 2)
    M = phi.matrix()
    assert M == DomainMatrix([[1, 0, -1, 0], [0, 0, -1, 1], [0, 1, -1, 0], [0, 0, -1, 0]], (4, 4), ZZ)