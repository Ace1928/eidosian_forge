from sympy.core.function import expand
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.mechanics import (PinJoint, JointsMethod, Body, KanesMethod,
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.testing.pytest import raises
from sympy.core.backend import zeros
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import solve
def test_complete_simple_double_pendulum():
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    m, l, g = symbols('m l g')
    C = Body('C')
    PartP = Body('P', mass=m)
    PartR = Body('R', mass=m)
    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1, child_point=-l * PartP.x, joint_axis=C.z)
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2, child_point=-l * PartR.x, joint_axis=PartP.z)
    PartP.apply_force(m * g * C.x)
    PartR.apply_force(m * g * C.x)
    method = JointsMethod(C, J1, J2)
    method.form_eoms()
    assert expand(method.mass_matrix_full) == Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2 * l ** 2 * m * cos(q2) + 3 * l ** 2 * m, l ** 2 * m * cos(q2) + l ** 2 * m], [0, 0, l ** 2 * m * cos(q2) + l ** 2 * m, l ** 2 * m]])
    assert trigsimp(method.forcing_full) == trigsimp(Matrix([[u1], [u2], [-g * l * m * (sin(q1 + q2) + sin(q1)) - g * l * m * sin(q1) + l ** 2 * m * (2 * u1 + u2) * u2 * sin(q2)], [-g * l * m * sin(q1 + q2) - l ** 2 * m * u1 ** 2 * sin(q2)]]))