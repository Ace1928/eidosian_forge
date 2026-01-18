from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_body_dcm():
    A = Body('A')
    B = Body('B')
    A.frame.orient_axis(B.frame, B.frame.z, 10)
    assert A.dcm(B) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])
    assert A.dcm(B.frame) == Matrix([[cos(10), sin(10), 0], [-sin(10), cos(10), 0], [0, 0, 1]])