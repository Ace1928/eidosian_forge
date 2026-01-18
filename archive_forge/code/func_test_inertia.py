from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_inertia():
    N = ReferenceFrame('N')
    ixx, iyy, izz = symbols('ixx iyy izz')
    ixy, iyz, izx = symbols('ixy iyz izx')
    assert inertia(N, ixx, iyy, izz) == ixx * (N.x | N.x) + iyy * (N.y | N.y) + izz * (N.z | N.z)
    assert inertia(N, 0, 0, 0) == 0 * (N.x | N.x)
    raises(TypeError, lambda: inertia(0, 0, 0, 0))
    assert inertia(N, ixx, iyy, izz, ixy, iyz, izx) == ixx * (N.x | N.x) + ixy * (N.x | N.y) + izx * (N.x | N.z) + ixy * (N.y | N.x) + iyy * (N.y | N.y) + iyz * (N.y | N.z) + izx * (N.z | N.x) + iyz * (N.z | N.y) + izz * (N.z | N.z)