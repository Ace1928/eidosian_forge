from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_Poly():
    K = FiniteExtension(Poly(x ** 2 - 2))
    p = Poly(x, y, domain=K)
    assert p.domain == K
    assert p.as_expr() == x
    assert (p ** 2).as_expr() == 2
    K = FiniteExtension(Poly(x ** 2 - 2, x, domain=QQ))
    K2 = FiniteExtension(Poly(t ** 2 - 2, t, domain=K))
    assert str(K2) == 'QQ[x]/(x**2 - 2)[t]/(t**2 - 2)'
    eK = K2.convert(x + t)
    assert K2.to_sympy(eK) == x + t
    assert K2.to_sympy(eK ** 2) == 4 + 2 * x * t
    p = Poly(x + t, y, domain=K2)
    assert p ** 2 == Poly(4 + 2 * x * t, y, domain=K2)