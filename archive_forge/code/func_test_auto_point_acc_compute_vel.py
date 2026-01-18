from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_point_acc_compute_vel():
    t = dynamicsymbols._t
    q1 = dynamicsymbols('q1')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, q1)
    O = Point('O')
    O.set_vel(N, 0)
    P = Point('P')
    P.set_pos(O, A.x)
    assert P.acc(N) == -q1.diff(t) ** 2 * A.x + q1.diff(t, 2) * A.y