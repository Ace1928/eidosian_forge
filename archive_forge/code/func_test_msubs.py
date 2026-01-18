from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_msubs():
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    expr = Matrix([[a * x + b, x * y.diff() + y], [x.diff().diff(), z + sin(z.diff())]])
    sol = Matrix([[a + b, y], [x.diff().diff(), 1]])
    sd = {x: 1, z: 1, z.diff(): 0, y.diff(): 0}
    assert msubs(expr, sd) == sol
    expr = cos(x + y) * tan(x + y) + b * x.diff()
    sd = {x: 0, y: pi / 2, x.diff(): 1}
    assert msubs(expr, sd, smart=True) == b + 1
    N = ReferenceFrame('N')
    v = x * N.x + y * N.y
    d = x * (N.x | N.x) + y * (N.y | N.y)
    v_sol = 1 * N.y
    d_sol = 1 * (N.y | N.y)
    sd = {x: 0, y: 1}
    assert msubs(v, sd) == v_sol
    assert msubs(d, sd) == d_sol