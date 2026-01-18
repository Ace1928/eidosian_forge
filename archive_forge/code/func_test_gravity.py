from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_gravity():
    N = ReferenceFrame('N')
    m, M, g = symbols('m M g')
    F1, F2 = dynamicsymbols('F1 F2')
    po = Point('po')
    pa = Particle('pa', po, m)
    A = ReferenceFrame('A')
    P = Point('P')
    I = outer(A.x, A.x)
    B = RigidBody('B', P, A, M, (I, P))
    forceList = [(po, F1), (P, F2)]
    forceList.extend(gravity(g * N.y, pa, B))
    l = [(po, F1), (P, F2), (po, g * m * N.y), (P, g * M * N.y)]
    for i in range(len(l)):
        for j in range(len(l[i])):
            assert forceList[i][j] == l[i][j]