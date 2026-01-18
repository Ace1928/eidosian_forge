from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_vel_multiple_path_warning_msg():
    N = ReferenceFrame('N')
    O = Point('O')
    P = Point('P')
    Q = Point('Q')
    P.set_vel(N, N.x)
    Q.set_vel(N, N.y)
    O.set_pos(P, N.z)
    O.set_pos(Q, N.y)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        O.vel(N)
        assert issubclass(w[-1].category, UserWarning)
        assert 'Velocity automatically calculated based on point' in str(w[-1].message)
        assert 'Velocities from these points are not necessarily the same. This may cause errors in your calculations.' in str(w[-1].message)