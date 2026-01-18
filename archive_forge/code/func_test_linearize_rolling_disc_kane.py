from sympy.core.backend import (symbols, Matrix, cos, sin, atan, sqrt,
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
from sympy.testing.pytest import slow
@slow
def test_linearize_rolling_disc_kane():
    t, r, m, g, v = symbols('t r m g v')
    q1, q2, q3, q4, q5, q6 = q = dynamicsymbols('q1:7')
    q1d, q2d, q3d, q4d, q5d, q6d = qd = [qi.diff(t) for qi in q]
    u = dynamicsymbols('u:6')
    u1, u2, u3, u4, u5, u6 = u = dynamicsymbols('u1:7')
    u1d, u2d, u3d, u4d, u5d, u6d = [ui.diff(t) for ui in u]
    N = ReferenceFrame('N')
    NO = Point('NO')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    CO = NO.locatenew('CO', q4 * N.x + q5 * N.y + q6 * N.z)
    w_c_n_qd = C.ang_vel_in(N)
    w_b_n_qd = B.ang_vel_in(N)
    C.set_ang_vel(N, u1 * B.x + u2 * B.y + u3 * B.z)
    v_co_n_qd = CO.pos_from(NO).dt(N)
    CO.set_vel(N, u4 * C.x + u5 * C.y + u6 * C.z)
    P = CO.locatenew('P', r * B.z)
    P.v2pt_theory(CO, N, C)
    f_c = Matrix([q6 - dot(CO.pos_from(P), N.z)])
    f_v = Matrix([dot(P.vel(N), uv) for uv in C])
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] + [dot(v_co_n_qd - CO.vel(N), uv) for uv in N])
    qdots = solve(kindiffs, qd)
    B.set_ang_vel(N, w_b_n_qd.subs(qdots))
    C.set_ang_acc(N, C.ang_vel_in(N).dt(B) + cross(B.ang_vel_in(N), C.ang_vel_in(N)))
    F_CO = m * g * A.z
    I = m * r ** 2 / 4
    J = m * r ** 2 / 2
    I_C_CO = inertia(C, I, J, I)
    Disc = RigidBody('Disc', CO, C, m, (I_C_CO, CO))
    BL = [Disc]
    FL = [(CO, F_CO)]
    KM = KanesMethod(N, [q1, q2, q3, q4, q5], [u1, u2, u3], kd_eqs=kindiffs, q_dependent=[q6], configuration_constraints=f_c, u_dependent=[u4, u5, u6], velocity_constraints=f_v)
    fr, fr_star = KM.kanes_equations(BL, FL)
    linearizer = KM.to_linearizer()
    assert linearizer.f_c == f_c
    assert linearizer.f_v == f_v
    assert linearizer.f_a == f_v.diff(t).subs(KM.kindiffdict())
    sol = solve(linearizer.f_0 + linearizer.f_1, qd)
    for qi in qdots.keys():
        assert sol[qi] == qdots[qi]
    assert simplify(linearizer.f_2 + linearizer.f_3 - fr - fr_star) == Matrix([0, 0, 0])
    q_op = {q6: -r * cos(q2)}
    u_op = {u1: 0, u2: sin(q2) * q1d + q3d, u3: cos(q2) * q1d, u4: -r * (sin(q2) * q1d + q3d) * cos(q3), u5: 0, u6: -r * (sin(q2) * q1d + q3d) * sin(q3)}
    qd_op = {q2d: 0, q4d: -r * (sin(q2) * q1d + q3d) * cos(q1), q5d: -r * (sin(q2) * q1d + q3d) * sin(q1), q6d: 0}
    ud_op = {u1d: 4 * g * sin(q2) / (5 * r) + sin(2 * q2) * q1d ** 2 / 2 + 6 * cos(q2) * q1d * q3d / 5, u2d: 0, u3d: 0, u4d: r * (sin(q2) * sin(q3) * q1d * q3d + sin(q3) * q3d ** 2), u5d: r * (4 * g * sin(q2) / (5 * r) + sin(2 * q2) * q1d ** 2 / 2 + 6 * cos(q2) * q1d * q3d / 5), u6d: -r * (sin(q2) * cos(q3) * q1d * q3d + cos(q3) * q3d ** 2)}
    A, B = linearizer.linearize(op_point=[q_op, u_op, qd_op, ud_op], A_and_B=True, simplify=True)
    upright_nominal = {q1d: 0, q2: 0, m: 1, r: 1, g: 1}
    A_sol = Matrix([[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [sin(q1) * q3d, 0, 0, 0, 0, -sin(q1), -cos(q1), 0], [-cos(q1) * q3d, 0, 0, 0, 0, cos(q1), -sin(q1), 0], [0, Rational(4, 5), 0, 0, 0, 0, 0, 6 * q3d / 5], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -2 * q3d, 0, 0]])
    B_sol = Matrix([])
    assert A.subs(upright_nominal) == A_sol
    assert B.subs(upright_nominal) == B_sol
    assert sympify(A.subs(upright_nominal).subs(q3d, 1 / sqrt(3))).eigenvals() == {0: 8}