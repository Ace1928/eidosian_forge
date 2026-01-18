from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_custom_rigid_body():
    rigidbody_masscenter = Point('rigidbody_masscenter')
    rigidbody_mass = Symbol('rigidbody_mass')
    rigidbody_frame = ReferenceFrame('rigidbody_frame')
    body_inertia = inertia(rigidbody_frame, 1, 0, 0)
    rigid_body = Body('rigidbody_body', rigidbody_masscenter, rigidbody_mass, rigidbody_frame, body_inertia)
    com = rigid_body.masscenter
    frame = rigid_body.frame
    rigidbody_masscenter.set_vel(rigidbody_frame, 0)
    assert com.vel(frame) == rigidbody_masscenter.vel(frame)
    assert com.pos_from(com) == rigidbody_masscenter.pos_from(com)
    assert rigid_body.mass == rigidbody_mass
    assert rigid_body.inertia == (body_inertia, rigidbody_masscenter)
    assert rigid_body.is_rigidbody
    assert hasattr(rigid_body, 'masscenter')
    assert hasattr(rigid_body, 'mass')
    assert hasattr(rigid_body, 'frame')
    assert hasattr(rigid_body, 'inertia')