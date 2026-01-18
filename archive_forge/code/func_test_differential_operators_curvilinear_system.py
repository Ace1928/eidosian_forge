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
def test_differential_operators_curvilinear_system():
    A = CoordSys3D('A', transformation='spherical', variable_names=['r', 'theta', 'phi'])
    B = CoordSys3D('B', transformation='cylindrical', variable_names=['r', 'theta', 'z'])
    assert gradient(3 * A.r + 4 * A.theta) == 3 * A.i + 4 / A.r * A.j
    assert gradient(3 * A.r * A.phi + 4 * A.theta) == 3 * A.phi * A.i + 4 / A.r * A.j + 3 / sin(A.theta) * A.k
    assert gradient(0 * A.r + 0 * A.theta + 0 * A.phi) == Vector.zero
    assert gradient(A.r * A.theta * A.phi) == A.theta * A.phi * A.i + A.phi * A.j + A.theta / sin(A.theta) * A.k
    assert divergence(A.r * A.i + A.theta * A.j + A.phi * A.k) == (sin(A.theta) * A.r + cos(A.theta) * A.r * A.theta) / (sin(A.theta) * A.r ** 2) + 3 + 1 / (sin(A.theta) * A.r)
    assert divergence(3 * A.r * A.phi * A.i + A.theta * A.j + A.r * A.theta * A.phi * A.k) == (sin(A.theta) * A.r + cos(A.theta) * A.r * A.theta) / (sin(A.theta) * A.r ** 2) + 9 * A.phi + A.theta / sin(A.theta)
    assert divergence(Vector.zero) == 0
    assert divergence(0 * A.i + 0 * A.j + 0 * A.k) == 0
    assert curl(A.r * A.i + A.theta * A.j + A.phi * A.k) == cos(A.theta) * A.phi / (sin(A.theta) * A.r) * A.i + -A.phi / A.r * A.j + A.theta / A.r * A.k
    assert curl(A.r * A.j + A.phi * A.k) == cos(A.theta) * A.phi / (sin(A.theta) * A.r) * A.i + -A.phi / A.r * A.j + 2 * A.k
    assert gradient(0 * B.r + 0 * B.theta + 0 * B.z) == Vector.zero
    assert gradient(B.r * B.theta * B.z) == B.theta * B.z * B.i + B.z * B.j + B.r * B.theta * B.k
    assert gradient(3 * B.r) == 3 * B.i
    assert gradient(2 * B.theta) == 2 / B.r * B.j
    assert gradient(4 * B.z) == 4 * B.k
    assert divergence(B.r * B.i + B.theta * B.j + B.z * B.k) == 3 + 1 / B.r
    assert divergence(B.r * B.j + B.z * B.k) == 1
    assert curl(B.r * B.j + B.z * B.k) == 2 * B.k
    assert curl(3 * B.i + 2 / B.r * B.j + 4 * B.k) == Vector.zero