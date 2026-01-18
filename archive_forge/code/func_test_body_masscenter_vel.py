from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_body_masscenter_vel():
    A = Body('A')
    N = ReferenceFrame('N')
    B = Body('B', frame=N)
    A.masscenter.set_vel(N, N.z)
    assert A.masscenter_vel(B) == N.z
    assert A.masscenter_vel(N) == N.z