from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, outer
from sympy.physics.vector.dyadic import _check_dyadic
from sympy.testing.pytest import raises
def test_dyadic_evalf():
    N = ReferenceFrame('N')
    a = pi * (N.x | N.x)
    assert a.evalf(3) == Float('3.1416', 3) * (N.x | N.x)
    s = symbols('s')
    a = 5 * s * pi * (N.x | N.x)
    assert a.evalf(2) == Float('5', 2) * Float('3.1416', 2) * s * (N.x | N.x)
    assert a.evalf(9, subs={s: 5.124}) == Float('80.48760378', 9) * (N.x | N.x)