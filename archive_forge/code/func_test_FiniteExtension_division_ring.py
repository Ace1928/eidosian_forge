from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension_division_ring():
    KQ = FiniteExtension(Poly(x ** 2 - 1, x, domain=QQ))
    KZ = FiniteExtension(Poly(x ** 2 - 1, x, domain=ZZ))
    KQt = FiniteExtension(Poly(x ** 2 - 1, x, domain=QQ[t]))
    KQtf = FiniteExtension(Poly(x ** 2 - 1, x, domain=QQ.frac_field(t)))
    assert KQ.is_Field is True
    assert KZ.is_Field is False
    assert KQt.is_Field is False
    assert KQtf.is_Field is True
    for K in (KQ, KZ, KQt, KQtf):
        xK = K.convert(x)
        assert xK / K.one == xK
        assert xK // K.one == xK
        assert xK % K.one == K.zero
        raises(ZeroDivisionError, lambda: xK / K.zero)
        raises(ZeroDivisionError, lambda: xK // K.zero)
        raises(ZeroDivisionError, lambda: xK % K.zero)
        if K.is_Field:
            assert xK / xK == K.one
            assert xK // xK == K.one
            assert xK % xK == K.zero
        else:
            raises(NotImplementedError, lambda: xK / xK)
            raises(NotImplementedError, lambda: xK // xK)
            raises(NotImplementedError, lambda: xK % xK)