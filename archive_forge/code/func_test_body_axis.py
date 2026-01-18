from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_body_axis():
    N = ReferenceFrame('N')
    B = Body('B', frame=N)
    assert B.x == N.x
    assert B.y == N.y
    assert B.z == N.z