from sympy.core.backend import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import (cross, dot, dynamicsymbols,
def test_aux_dep():
    t, r, m, g, I, J = symbols('t r m g I J')
    Fx, Fy, Fz = symbols('Fx Fy Fz')
    q = dynamicsymbols('q:4')
    qd = [qi.diff(t) for qi in q]
    u = dynamicsymbols('u:6')
    ud = [ui.diff(t) for ui in u]
    ud_zero = dict(zip(ud, [0.0] * len(ud)))
    ua = dynamicsymbols('ua:3')
    ua_zero = dict(zip(ua, [0.0] * len(ua)))
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q[0], N.z])
    B = A.orientnew('B', 'Axis', [q[1], A.x])
    C = B.orientnew('C', 'Axis', [q[2], B.y])
    C.set_ang_vel(N, u[0] * B.x + u[1] * B.y + u[2] * B.z)
    C.set_ang_acc(N, C.ang_vel_in(N).diff(t, B) + cross(B.ang_vel_in(N), C.ang_vel_in(N)))
    P = Point('P')
    P.set_vel(N, ua[0] * A.x + ua[1] * A.y + ua[2] * A.z)
    O = P.locatenew('O', q[3] * A.z + r * sin(q[1]) * A.y)
    O.set_vel(N, u[3] * A.x + u[4] * A.y + u[5] * A.z)
    O.set_acc(N, O.vel(N).diff(t, A) + cross(A.ang_vel_in(N), O.vel(N)))
    w_c_n_qd = qd[0] * A.z + qd[1] * B.x + qd[2] * B.y
    v_o_n_qd = O.pos_from(P).diff(t, A) + cross(A.ang_vel_in(N), O.pos_from(P))
    kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] + [dot(v_o_n_qd - O.vel(N), A.z)])
    qd_kd = solve(kindiffs, qd)
    steady_conditions = solve(kindiffs.subs({qd[1]: 0, qd[3]: 0}), u)
    steady_conditions.update({qd[1]: 0, qd[3]: 0})
    partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u + ua]
    partial_v_O = [O.vel(N).diff(ui, N) for ui in u + ua]
    partial_v_P = [P.vel(N).diff(ui, N) for ui in u + ua]
    f_c = Matrix([dot(-r * B.z, A.z) - q[3]])
    f_v = Matrix([dot(O.vel(N) - (P.vel(N) + cross(C.ang_vel_in(N), O.pos_from(P))), ai).expand() for ai in A])
    v_o_n = cross(C.ang_vel_in(N), O.pos_from(P))
    a_o_n = v_o_n.diff(t, A) + cross(A.ang_vel_in(N), v_o_n)
    f_a = Matrix([dot(O.acc(N) - a_o_n, ai) for ai in A])
    M_v = zeros(3, 9)
    for i in range(3):
        for j, ui in enumerate(u + ua):
            M_v[i, j] = f_v[i].diff(ui)
    M_v_i = M_v[:, :3]
    M_v_d = M_v[:, 3:6]
    M_v_aux = M_v[:, 6:]
    M_v_i_aux = M_v_i.row_join(M_v_aux)
    A_rs = -M_v_d.inv() * M_v_i_aux
    u_dep = A_rs[:, :3] * Matrix(u[:3])
    u_dep_dict = dict(zip(u[3:], u_dep))
    F_O = m * g * A.z
    F_P = Fx * A.x + Fy * A.y + Fz * A.z
    Fr_u = Matrix([dot(F_O, pv_o) + dot(F_P, pv_p) for pv_o, pv_p in zip(partial_v_O, partial_v_P)])
    R_star_O = -m * O.acc(N)
    I_C_O = inertia(B, I, J, I)
    T_star_C = -(dot(I_C_O, C.ang_acc_in(N)) + cross(C.ang_vel_in(N), dot(I_C_O, C.ang_vel_in(N))))
    Fr_star_u = Matrix([dot(R_star_O, pv) + dot(T_star_C, pav) for pv, pav in zip(partial_v_O, partial_w_C)])
    Fr_c = Fr_u[:3, :].col_join(Fr_u[6:, :]) + A_rs.T * Fr_u[3:6, :]
    Fr_star_c = Fr_star_u[:3, :].col_join(Fr_star_u[6:, :]) + A_rs.T * Fr_star_u[3:6, :]
    Fr_star_steady = Fr_star_c.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions).subs({q[3]: -r * cos(q[1])}).expand()
    iner_tuple = (I_C_O, O)
    disc = RigidBody('disc', O, C, m, iner_tuple)
    bodyList = [disc]
    F_o = (O, F_O)
    F_p = (P, F_P)
    forceList = [F_o, F_p]
    kane = KanesMethod(N, q_ind=q[:3], u_ind=u[:3], kd_eqs=kindiffs, q_dependent=q[3:], configuration_constraints=f_c, u_dependent=u[3:], velocity_constraints=f_v, u_auxiliary=ua)
    fr, frstar = kane.kanes_equations(bodyList, forceList)
    frstar_steady = frstar.subs(ud_zero).subs(u_dep_dict).subs(steady_conditions).subs({q[3]: -r * cos(q[1])}).expand()
    kdd = kane.kindiffdict()
    assert Matrix(Fr_c).expand() == fr.expand()
    assert Matrix(Fr_star_c.subs(kdd)).expand() == frstar.expand()
    assert simplify(Matrix(Fr_star_steady).expand()) == simplify(frstar_steady.expand())
    syms_in_forcing = find_dynamicsymbols(kane.forcing)
    for qdi in qd:
        assert qdi not in syms_in_forcing