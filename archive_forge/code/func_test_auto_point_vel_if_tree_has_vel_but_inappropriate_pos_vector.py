from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_point_vel_if_tree_has_vel_but_inappropriate_pos_vector():
    q1, q2 = dynamicsymbols('q1 q2')
    B = ReferenceFrame('B')
    S = ReferenceFrame('S')
    P = Point('P')
    P.set_vel(B, q1 * B.x)
    P1 = Point('P1')
    P1.set_pos(P, S.y)
    raises(ValueError, lambda: P1.vel(B))
    raises(ValueError, lambda: P1.vel(S))