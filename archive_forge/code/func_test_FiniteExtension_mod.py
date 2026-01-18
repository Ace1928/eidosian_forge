from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_mod():
    K = FiniteExtension(Poly(x ** 3 + 1, x, domain=QQ))
    xf = K(x)
    assert (xf ** 2 - 1) % 1 == K.zero
    assert 1 % (xf ** 2 - 1) == K.zero
    assert (xf ** 2 - 1) / (xf - 1) == xf + 1
    assert (xf ** 2 - 1) // (xf - 1) == xf + 1
    assert (xf ** 2 - 1) % (xf - 1) == K.zero
    raises(ZeroDivisionError, lambda: (xf ** 2 - 1) % 0)
    raises(TypeError, lambda: xf % [])
    raises(TypeError, lambda: [] % xf)
    K = FiniteExtension(Poly(x ** 3 + 1, x, domain=ZZ))
    xf = K(x)
    assert (xf ** 2 - 1) % 1 == K.zero
    raises(NotImplementedError, lambda: (xf ** 2 - 1) % (xf - 1))