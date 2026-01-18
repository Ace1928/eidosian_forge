from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
from sympy.testing.pytest import raises
def test_dynamicsymbols():
    f1 = dynamicsymbols('f1')
    f2 = dynamicsymbols('f2', real=True)
    f3 = dynamicsymbols('f3', positive=True)
    f4, f5 = dynamicsymbols('f4,f5', commutative=False)
    f6 = dynamicsymbols('f6', integer=True)
    assert f1.is_real is None
    assert f2.is_real
    assert f3.is_positive
    assert f4 * f5 != f5 * f4
    assert f6.is_integer