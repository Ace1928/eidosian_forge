from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_set_domain():
    KZ = FiniteExtension(Poly(x ** 2 + 1, x, domain='ZZ'))
    KQ = FiniteExtension(Poly(x ** 2 + 1, x, domain='QQ'))
    assert KZ.set_domain(QQ) == KQ