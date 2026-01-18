from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_body_ang_vel():
    A = Body('A')
    N = ReferenceFrame('N')
    B = Body('B', frame=N)
    A.frame.set_ang_vel(N, N.y)
    assert A.ang_vel_in(B) == N.y
    assert B.ang_vel_in(A) == -N.y
    assert A.ang_vel_in(N) == N.y