from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_acc_derivative():
    q1, q2 = dynamicsymbols('q1:3')
    u1, u2 = dynamicsymbols('q1:3', 1)
    v1, v2 = dynamicsymbols('q1:3', 2)
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    B.orient_axis(A, A.z, q1)
    C.orient_axis(B, B.z, q2)
    Am = Point('Am')
    Am.set_vel(A, 0)
    Bm = Point('Bm')
    Bm.set_pos(Am, B.x)
    Bm.set_vel(B, 0)
    Bm.set_vel(C, 0)
    Cm = Point('Cm')
    Cm.set_pos(Bm, C.x)
    Cm.set_vel(C, 0)
    Bm_vel_dict, Cm_vel_dict = (Bm._vel_dict.copy(), Cm._vel_dict.copy())
    Bm_acc_dict, Cm_acc_dict = (Bm._acc_dict.copy(), Cm._acc_dict.copy())
    check = -u1 ** 2 * B.x + v1 * B.y - (u1 + u2) ** 2 * C.x + (v1 + v2) * C.y
    assert Cm.acc(A) == check
    Bm._vel_dict, Cm._vel_dict = (Bm_vel_dict, Cm_vel_dict)
    Bm._acc_dict, Cm._acc_dict = (Bm_acc_dict, Cm_acc_dict)
    Bm.v2pt_theory(Am, A, B)
    Cm.v2pt_theory(Bm, A, C)
    Bm.a2pt_theory(Am, A, B)
    assert Cm.a2pt_theory(Bm, A, C) == check