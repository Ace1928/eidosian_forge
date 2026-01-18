import sympy.physics.mechanics.models as models
from sympy.core.backend import (cos, sin, Matrix, symbols, zeros)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols)
def test_n_link_pendulum_on_cart_higher_order():
    l0, m0 = symbols('l0 m0')
    l1, m1 = symbols('l1 m1')
    m2 = symbols('m2')
    g = symbols('g')
    q0, q1, q2 = dynamicsymbols('q0 q1 q2')
    u0, u1, u2 = dynamicsymbols('u0 u1 u2')
    F, T1 = dynamicsymbols('F T1')
    kane1 = models.n_link_pendulum_on_cart(2)
    massmatrix1 = Matrix([[m0 + m1 + m2, -l0 * m1 * cos(q1) - l0 * m2 * cos(q1), -l1 * m2 * cos(q2)], [-l0 * m1 * cos(q1) - l0 * m2 * cos(q1), l0 ** 2 * m1 + l0 ** 2 * m2, l0 * l1 * m2 * (sin(q1) * sin(q2) + cos(q1) * cos(q2))], [-l1 * m2 * cos(q2), l0 * l1 * m2 * (sin(q1) * sin(q2) + cos(q1) * cos(q2)), l1 ** 2 * m2]])
    forcing1 = Matrix([[-l0 * m1 * u1 ** 2 * sin(q1) - l0 * m2 * u1 ** 2 * sin(q1) - l1 * m2 * u2 ** 2 * sin(q2) + F], [g * l0 * m1 * sin(q1) + g * l0 * m2 * sin(q1) - l0 * l1 * m2 * (sin(q1) * cos(q2) - sin(q2) * cos(q1)) * u2 ** 2], [g * l1 * m2 * sin(q2) - l0 * l1 * m2 * (-sin(q1) * cos(q2) + sin(q2) * cos(q1)) * u1 ** 2]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])