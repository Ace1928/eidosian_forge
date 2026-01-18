from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
from sympy.testing.pytest import raises
def test_mixed_coordinates():
    a = CoordSys3D('a')
    b = CoordSys3D('b')
    c = CoordSys3D('c')
    assert gradient(a.x * b.y) == b.y * a.i + a.x * b.j
    assert gradient(3 * cos(q) * a.x * b.x + a.y * (a.x + (cos(q) + b.x))) == (a.y + 3 * b.x * cos(q)) * a.i + (a.x + b.x + cos(q)) * a.j + (3 * a.x * cos(q) + a.y) * b.i
    assert gradient(a.x ** b.y) == Gradient(a.x ** b.y)
    assert gradient(cos(a.x * b.y)) == Gradient(cos(a.x * b.y))
    assert gradient(3 * cos(q) * a.x * b.x * a.z * a.y + b.y * b.z + cos(a.x + a.y) * b.z) == (3 * a.y * a.z * b.x * cos(q) - b.z * sin(a.x + a.y)) * a.i + (3 * a.x * a.z * b.x * cos(q) - b.z * sin(a.x + a.y)) * a.j + 3 * a.x * a.y * b.x * cos(q) * a.k + 3 * a.x * a.y * a.z * cos(q) * b.i + b.z * b.j + (b.y + cos(a.x + a.y)) * b.k
    assert divergence(a.i * a.x + a.j * a.y + a.z * a.k + b.i * b.x + b.j * b.y + b.z * b.k + c.i * c.x + c.j * c.y + c.z * c.k) == S(9)
    assert divergence(3 * a.i * a.x * a.z + b.j * b.x * c.z + 3 * a.j * a.z * a.y) == 6 * a.z + b.x * Dot(b.j, c.k)
    assert divergence(3 * cos(q) * a.x * b.x * b.i * c.x) == 3 * a.x * b.x * cos(q) * Dot(b.i, c.i) + 3 * a.x * c.x * cos(q) + 3 * b.x * c.x * cos(q) * Dot(b.i, a.i)
    assert divergence(a.x * b.x * c.x * Cross(a.x * a.i, a.y * b.j)) == a.x * b.x * c.x * Divergence(Cross(a.x * a.i, a.y * b.j)) + b.x * c.x * Dot(Cross(a.x * a.i, a.y * b.j), a.i) + a.x * c.x * Dot(Cross(a.x * a.i, a.y * b.j), b.i) + a.x * b.x * Dot(Cross(a.x * a.i, a.y * b.j), c.i)
    assert divergence(a.x * b.x * c.x * (a.x * a.i + b.x * b.i)) == 4 * a.x * b.x * c.x + a.x ** 2 * c.x * Dot(a.i, b.i) + a.x ** 2 * b.x * Dot(a.i, c.i) + b.x ** 2 * c.x * Dot(b.i, a.i) + a.x * b.x ** 2 * Dot(b.i, c.i)