from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test_disc_on_an_incline_plane():
    y, theta = dynamicsymbols('y theta')
    yd, thetad = dynamicsymbols('y theta', 1)
    m, g, R, l, alpha = symbols('m g R l alpha')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [pi / 2 - alpha, N.z])
    B = A.orientnew('B', 'Axis', [-theta, A.z])
    Do = Point('Do')
    Do.set_vel(N, yd * A.x)
    I = m * R ** 2 / 2 * B.z | B.z
    D = RigidBody('D', Do, B, m, (I, Do))
    D.potential_energy = m * g * (l - y) * sin(alpha)
    L = Lagrangian(N, D)
    q = [y, theta]
    hol_coneqs = [y - R * theta]
    m = LagrangesMethod(L, q, hol_coneqs=hol_coneqs)
    m.form_lagranges_equations()
    rhs = m.rhs()
    rhs.simplify()
    assert rhs[2] == 2 * g * sin(alpha) / 3