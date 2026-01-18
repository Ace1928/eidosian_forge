from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_from_sympy():
    K = FiniteExtension(Poly(x ** 3 + 1, x, domain=ZZ))
    xf = K(x)
    assert K.from_sympy(x) == xf
    assert K.to_sympy(xf) == x