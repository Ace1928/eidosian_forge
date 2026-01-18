from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_angular_momentum_and_linear_momentum():
    """A rod with length 2l, centroidal inertia I, and mass M along with a
    particle of mass m fixed to the end of the rod rotate with an angular rate
    of omega about point O which is fixed to the non-particle end of the rod.
    The rod's reference frame is A and the inertial frame is N."""
    m, M, l, I = symbols('m, M, l, I')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    a = ReferenceFrame('a')
    O = Point('O')
    Ac = O.locatenew('Ac', l * N.x)
    P = Ac.locatenew('P', l * N.x)
    O.set_vel(N, 0 * N.x)
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    A = RigidBody('A', Ac, a, M, (I * outer(N.z, N.z), Ac))
    expected = 2 * m * omega * l * N.y + M * l * omega * N.y
    assert linear_momentum(N, A, Pa) == expected
    raises(TypeError, lambda: angular_momentum(N, N, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, O, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, N, O, Pa))
    expected = (I + M * l ** 2 + 4 * m * l ** 2) * omega * N.z
    assert angular_momentum(O, N, A, Pa) == expected