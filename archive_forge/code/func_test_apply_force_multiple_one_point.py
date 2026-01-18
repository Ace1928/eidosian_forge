from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_apply_force_multiple_one_point():
    a, b = symbols('a b')
    P = Point('P')
    B = Body('B')
    f1 = a * B.x
    f2 = b * B.y
    B.apply_force(f1, P)
    assert B.loads == [(P, f1)]
    B.apply_force(f2, P)
    assert B.loads == [(P, f1 + f2)]