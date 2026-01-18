from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core.backend import Matrix, _simplify_matrix, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (dynamicsymbols, Body, JointsMethod,
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_cylindrical_joint():
    N, A, P, C = _generate_body()
    q0_def, q1_def, u0_def, u1_def = dynamicsymbols('q0:2_J, u0:2_J')
    Cj = CylindricalJoint('J', P, C)
    assert Cj.name == 'J'
    assert Cj.parent == P
    assert Cj.child == C
    assert Cj.coordinates == Matrix([q0_def, q1_def])
    assert Cj.speeds == Matrix([u0_def, u1_def])
    assert Cj.rotation_coordinate == q0_def
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0_def
    assert Cj.translation_speed == u1_def
    assert Cj.kdes == Matrix([u0_def - q0_def.diff(t), u1_def - q1_def.diff(t)])
    assert Cj.joint_axis == N.x
    assert Cj.child_point.pos_from(C.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(P.masscenter) == Vector(0)
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.x
    assert C.masscenter.pos_from(P.masscenter) == q1_def * N.x
    assert Cj.child_point.vel(N) == u1_def * N.x
    assert A.ang_vel_in(N) == u0_def * N.x
    assert Cj.parent_interframe == N
    assert Cj.child_interframe == A
    assert Cj.__str__() == 'CylindricalJoint: J  parent: P  child: C'
    q0, q1, u0, u1 = dynamicsymbols('q0:2, u0:2')
    l, m = symbols('l, m')
    N, A, P, C, Pint, Cint = _generate_body(True)
    Cj = CylindricalJoint('J', P, C, rotation_coordinate=q0, rotation_speed=u0, translation_speed=u1, parent_point=m * N.x, child_point=l * A.y, parent_interframe=Pint, child_interframe=Cint, joint_axis=2 * N.z)
    assert Cj.coordinates == Matrix([q0, q1_def])
    assert Cj.speeds == Matrix([u0, u1])
    assert Cj.rotation_coordinate == q0
    assert Cj.translation_coordinate == q1_def
    assert Cj.rotation_speed == u0
    assert Cj.translation_speed == u1
    assert Cj.kdes == Matrix([u0 - q0.diff(t), u1 - q1_def.diff(t)])
    assert Cj.joint_axis == 2 * N.z
    assert Cj.child_point.pos_from(C.masscenter) == l * A.y
    assert Cj.parent_point.pos_from(P.masscenter) == m * N.x
    assert Cj.parent_point.pos_from(Cj._child_point) == -q1_def * N.z
    assert C.masscenter.pos_from(P.masscenter) == m * N.x + q1_def * N.z - l * A.y
    assert C.masscenter.vel(N) == u1 * N.z - u0 * l * A.z
    assert A.ang_vel_in(N) == u0 * N.z