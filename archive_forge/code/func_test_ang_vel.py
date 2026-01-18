from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
def test_ang_vel():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    D = N.orientnew('D', 'Axis', [q4, N.y])
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    assert A.ang_vel_in(N) == q1d * A.z
    assert B.ang_vel_in(N) == q2d * B.x + q1d * A.z
    assert C.ang_vel_in(N) == q3d * C.y + q2d * B.x + q1d * A.z
    A2 = N.orientnew('A2', 'Axis', [q4, N.y])
    assert N.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == -q1d * N.z
    assert N.ang_vel_in(B) == -q1d * A.z - q2d * B.x
    assert N.ang_vel_in(C) == -q1d * A.z - q2d * B.x - q3d * B.y
    assert N.ang_vel_in(A2) == -q4d * N.y
    assert A.ang_vel_in(N) == q1d * N.z
    assert A.ang_vel_in(A) == 0
    assert A.ang_vel_in(B) == -q2d * B.x
    assert A.ang_vel_in(C) == -q2d * B.x - q3d * B.y
    assert A.ang_vel_in(A2) == q1d * N.z - q4d * N.y
    assert B.ang_vel_in(N) == q1d * A.z + q2d * A.x
    assert B.ang_vel_in(A) == q2d * A.x
    assert B.ang_vel_in(B) == 0
    assert B.ang_vel_in(C) == -q3d * B.y
    assert B.ang_vel_in(A2) == q1d * A.z + q2d * A.x - q4d * N.y
    assert C.ang_vel_in(N) == q1d * A.z + q2d * A.x + q3d * B.y
    assert C.ang_vel_in(A) == q2d * A.x + q3d * C.y
    assert C.ang_vel_in(B) == q3d * B.y
    assert C.ang_vel_in(C) == 0
    assert C.ang_vel_in(A2) == q1d * A.z + q2d * A.x + q3d * B.y - q4d * N.y
    assert A2.ang_vel_in(N) == q4d * A2.y
    assert A2.ang_vel_in(A) == q4d * A2.y - q1d * N.z
    assert A2.ang_vel_in(B) == q4d * N.y - q1d * A.z - q2d * A.x
    assert A2.ang_vel_in(C) == q4d * N.y - q1d * A.z - q2d * A.x - q3d * B.y
    assert A2.ang_vel_in(A2) == 0
    C.set_ang_vel(N, u1 * C.x + u2 * C.y + u3 * C.z)
    assert C.ang_vel_in(N) == u1 * C.x + u2 * C.y + u3 * C.z
    assert N.ang_vel_in(C) == -u1 * C.x + -u2 * C.y + -u3 * C.z
    assert C.ang_vel_in(D) == u1 * C.x + u2 * C.y + u3 * C.z + -q4d * D.y
    assert D.ang_vel_in(C) == -u1 * C.x + -u2 * C.y + -u3 * C.z + q4d * D.y
    q0 = dynamicsymbols('q0')
    q0d = dynamicsymbols('q0', 1)
    E = N.orientnew('E', 'Quaternion', (q0, q1, q2, q3))
    assert E.ang_vel_in(N) == 2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1) * E.x + 2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2) * E.y + 2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3) * E.z
    F = N.orientnew('F', 'Body', (q1, q2, q3), 313)
    assert F.ang_vel_in(N) == (sin(q2) * sin(q3) * q1d + cos(q3) * q2d) * F.x + (sin(q2) * cos(q3) * q1d - sin(q3) * q2d) * F.y + (cos(q2) * q1d + q3d) * F.z
    G = N.orientnew('G', 'Axis', (q1, N.x + N.y))
    assert G.ang_vel_in(N) == q1d * (N.x + N.y).normalize()
    assert N.ang_vel_in(G) == -q1d * (N.x + N.y).normalize()