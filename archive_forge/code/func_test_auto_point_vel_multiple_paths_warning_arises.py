from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_point_vel_multiple_paths_warning_arises():
    q, u = dynamicsymbols('q u')
    N = ReferenceFrame('N')
    O = Point('O')
    P = Point('P')
    Q = Point('Q')
    R = Point('R')
    P.set_vel(N, u * N.x)
    Q.set_vel(N, u * N.y)
    R.set_vel(N, u * N.z)
    O.set_pos(P, q * N.z)
    O.set_pos(Q, q * N.y)
    O.set_pos(R, q * N.x)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        raises(UserWarning, lambda: O.vel(N))