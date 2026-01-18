from sympy.core.backend import (symbols, Matrix, cos, sin, atan, sqrt,
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
from sympy.testing.pytest import slow
def test_linearize_pendulum_kane_minimal():
    q1 = dynamicsymbols('q1')
    u1 = dynamicsymbols('u1')
    q1d = dynamicsymbols('q1', 1)
    L, m, t = symbols('L, m, t')
    g = 9.8
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, u1 * N.z)
    P = pN.locatenew('P', L * A.x)
    P.v2pt_theory(pN, N, A)
    pP = Particle('pP', P, m)
    kde = Matrix([q1d - u1])
    R = m * g * N.x
    KM = KanesMethod(N, q_ind=[q1], u_ind=[u1], kd_eqs=kde)
    fr, frstar = KM.kanes_equations([pP], [(P, R)])
    A, B, inp_vec = KM.linearize(A_and_B=True, simplify=True)
    assert A == Matrix([[0, 1], [-9.8 * cos(q1) / L, 0]])
    assert B == Matrix([])