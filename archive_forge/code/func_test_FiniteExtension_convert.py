from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_convert():
    K1 = FiniteExtension(Poly(x ** 2 + 1))
    K2 = QQ[x]
    x1, x2 = (K1(x), K2(x))
    assert K1.convert(x2) == x1
    assert K2.convert(x1) == x2
    K = FiniteExtension(Poly(x ** 2 - 1, domain=QQ))
    assert K.convert_from(QQ(1, 2), QQ) == K.one / 2