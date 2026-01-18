from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, outer
from sympy.physics.vector.dyadic import _check_dyadic
from sympy.testing.pytest import raises
def test_dyadic_subs():
    N = ReferenceFrame('N')
    s = symbols('s')
    a = s * (N.x | N.x)
    assert a.subs({s: 2}) == 2 * (N.x | N.x)