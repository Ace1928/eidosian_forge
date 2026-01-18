from sympy import solve
from sympy.core.backend import (cos, expand, Matrix, sin, symbols, tan, sqrt, S,
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.testing.pytest import raises
from sympy.core.backend import USE_SYMENGINE
def test_implicit_kinematics():
    NED = ReferenceFrame('NED')
    NED_o = Point('NED_o')
    NED_o.set_vel(NED, 0)
    q_att = dynamicsymbols('lambda_0:4', real=True)
    B = NED.orientnew('B', 'Quaternion', q_att)
    q_pos = dynamicsymbols('B_x:z')
    B_cm = NED_o.locatenew('B_cm', q_pos[0] * B.x + q_pos[1] * B.y + q_pos[2] * B.z)
    q_ind = q_att[1:] + q_pos
    q_dep = [q_att[0]]
    kinematic_eqs = []
    B_ang_vel = B.ang_vel_in(NED)
    P, Q, R = dynamicsymbols('P Q R')
    B.set_ang_vel(NED, P * B.x + Q * B.y + R * B.z)
    B_ang_vel_kd = (B.ang_vel_in(NED) - B_ang_vel).simplify()
    kinematic_eqs += [B_ang_vel_kd & B.x, B_ang_vel_kd & B.y, B_ang_vel_kd & B.z]
    B_cm_vel = B_cm.vel(NED)
    U, V, W = dynamicsymbols('U V W')
    B_cm.set_vel(NED, U * B.x + V * B.y + W * B.z)
    B_ref_vel_kd = B_cm.vel(NED) - B_cm_vel
    kinematic_eqs += [B_ref_vel_kd & B.x, B_ref_vel_kd & B.y, B_ref_vel_kd & B.z]
    u_ind = [U, V, W, P, Q, R]
    q_att_vec = Matrix(q_att)
    config_cons = [(q_att_vec.T * q_att_vec)[0] - 1]
    kinematic_eqs = kinematic_eqs + [(q_att_vec.T * q_att_vec.diff())[0]]
    try:
        KM = KanesMethod(NED, q_ind, u_ind, q_dependent=q_dep, kd_eqs=kinematic_eqs, configuration_constraints=config_cons, velocity_constraints=[], u_dependent=[], u_auxiliary=[], explicit_kinematics=False)
    except Exception as e:
        if USE_SYMENGINE and 'Matrix is rank deficient' in str(e):
            return
        else:
            raise e
    M_B = symbols('M_B')
    J_B = inertia(B, *[S(f'J_B_{ax}') * (1 if ax[0] == ax[1] else -1) for ax in ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']])
    J_B = J_B.subs({S('J_B_xy'): 0, S('J_B_yz'): 0})
    RB = RigidBody('RB', B_cm, B, M_B, (J_B, B_cm))
    rigid_bodies = [RB]
    force_list = [(RB.masscenter, RB.mass * S('g') * NED.z), (RB.frame, dynamicsymbols('T_z') * B.z), (RB.masscenter, dynamicsymbols('F_z') * B.z)]
    KM.kanes_equations(rigid_bodies, force_list)
    n_ops_implicit = sum([x.count_ops() for x in KM.forcing_full] + [x.count_ops() for x in KM.mass_matrix_full])
    mass_matrix_kin_implicit = KM.mass_matrix_kin
    forcing_kin_implicit = KM.forcing_kin
    KM.explicit_kinematics = True
    n_ops_explicit = sum([x.count_ops() for x in KM.forcing_full] + [x.count_ops() for x in KM.mass_matrix_full])
    forcing_kin_explicit = KM.forcing_kin
    assert n_ops_implicit / n_ops_explicit < 0.05
    assert mass_matrix_kin_implicit * KM.q.diff() - forcing_kin_implicit == Matrix(kinematic_eqs)
    qdot_candidate = forcing_kin_explicit
    quat_dot_textbook = Matrix([[0, -P, -Q, -R], [P, 0, R, -Q], [Q, -R, 0, P], [R, Q, -P, 0]]) * q_att_vec / 2
    qdot_candidate[-1] = quat_dot_textbook[0]
    qdot_candidate[0] = quat_dot_textbook[1]
    qdot_candidate[1] = quat_dot_textbook[2]
    qdot_candidate[2] = quat_dot_textbook[3]
    lambda_0_sol = solve(config_cons[0], q_att_vec[0])[1]
    lhs_candidate = simplify(mass_matrix_kin_implicit * qdot_candidate).subs({q_att_vec[0]: lambda_0_sol})
    assert lhs_candidate == forcing_kin_implicit