from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.physics.vector import ReferenceFrame, Vector, Point, \
from sympy.physics.vector.fieldfunctions import divergence, \
from sympy.testing.pytest import raises
def test_gradient():
    a = Symbol('a')
    assert gradient(0, R) == Vector(0)
    assert gradient(R[0], R) == R.x
    assert gradient(R[0] * R[1] * R[2], R) == R[1] * R[2] * R.x + R[0] * R[2] * R.y + R[0] * R[1] * R.z
    assert gradient(2 * R[0] ** 2, R) == 4 * R[0] * R.x
    assert gradient(a * sin(R[1]) / R[0], R) == -a * sin(R[1]) / R[0] ** 2 * R.x + a * cos(R[1]) / R[0] * R.y
    assert gradient(P[0] * P[1], R) == ((-R[0] * sin(q) + R[1] * cos(q)) * cos(q) - (R[0] * cos(q) + R[1] * sin(q)) * sin(q)) * R.x + ((-R[0] * sin(q) + R[1] * cos(q)) * sin(q) + (R[0] * cos(q) + R[1] * sin(q)) * cos(q)) * R.y
    assert gradient(P[0] * R[2], P) == P[2] * P.x + P[0] * P.z