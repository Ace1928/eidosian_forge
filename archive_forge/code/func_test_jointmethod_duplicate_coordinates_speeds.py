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
def test_jointmethod_duplicate_coordinates_speeds():
    P = Body('P')
    C = Body('C')
    T = Body('T')
    q, u = dynamicsymbols('q u')
    P1 = PinJoint('P1', P, C, q)
    P2 = PrismaticJoint('P2', C, T, q)
    raises(ValueError, lambda: JointsMethod(P, P1, P2))
    P1 = PinJoint('P1', P, C, speeds=u)
    P2 = PrismaticJoint('P2', C, T, speeds=u)
    raises(ValueError, lambda: JointsMethod(P, P1, P2))
    P1 = PinJoint('P1', P, C, q, u)
    P2 = PrismaticJoint('P2', C, T, q, u)
    raises(ValueError, lambda: JointsMethod(P, P1, P2))