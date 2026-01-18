from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_clear_load():
    a = symbols('a')
    P = Point('P')
    B = Body('B')
    force = a * B.z
    B.apply_force(force, P)
    assert B.loads == [(P, force)]
    B.clear_loads()
    assert B.loads == []