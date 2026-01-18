from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_body_add_torque():
    body = Body('body')
    torque_vector = body.frame.x
    body.apply_torque(torque_vector)
    assert len(body.loads) == 1
    assert body.loads[0] == (body.frame, torque_vector)
    raises(TypeError, lambda: body.apply_torque(0))