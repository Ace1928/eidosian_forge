from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_vel_cyclic_warning_arises():
    P = Point('P')
    P1 = Point('P1')
    P2 = Point('P2')
    P3 = Point('P3')
    N = ReferenceFrame('N')
    P.set_vel(N, N.x)
    P1.set_pos(P, N.x)
    P2.set_pos(P1, N.y)
    P3.set_pos(P2, N.z)
    P1.set_pos(P3, N.x + N.y)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        raises(UserWarning, lambda: P2.vel(N))