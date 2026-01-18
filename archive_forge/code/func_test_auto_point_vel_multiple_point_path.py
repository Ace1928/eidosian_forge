from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_point_vel_multiple_point_path():
    t = dynamicsymbols._t
    q1, q2 = dynamicsymbols('q1 q2')
    B = ReferenceFrame('B')
    P = Point('P')
    P.set_vel(B, q1 * B.x)
    P1 = Point('P1')
    P1.set_pos(P, q2 * B.y)
    P1.set_vel(B, q1 * B.z)
    P2 = Point('P2')
    P2.set_pos(P1, q1 * B.z)
    P3 = Point('P3')
    P3.set_pos(P2, 10 * q1 * B.y)
    assert P3.vel(B) == 10 * q1.diff(t) * B.y + (q1 + q1.diff(t)) * B.z