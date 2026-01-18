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
def test_del_operator():
    assert delop ^ Vector.zero == Vector.zero
    assert (delop ^ Vector.zero).doit() == Vector.zero == curl(Vector.zero)
    assert delop.cross(Vector.zero) == delop ^ Vector.zero
    assert (delop ^ i).doit() == Vector.zero
    assert delop.cross(2 * y ** 2 * j, doit=True) == Vector.zero
    assert delop.cross(2 * y ** 2 * j) == delop ^ 2 * y ** 2 * j
    v = x * y * z * (i + j + k)
    assert (delop ^ v).doit() == (-x * y + x * z) * i + (x * y - y * z) * j + (-x * z + y * z) * k == curl(v)
    assert delop ^ v == delop.cross(v)
    assert delop.cross(2 * x ** 2 * j) == (Derivative(0, C.y) - Derivative(2 * C.x ** 2, C.z)) * C.i + (-Derivative(0, C.x) + Derivative(0, C.z)) * C.j + (-Derivative(0, C.y) + Derivative(2 * C.x ** 2, C.x)) * C.k
    assert delop.cross(2 * x ** 2 * j, doit=True) == 4 * x * k == curl(2 * x ** 2 * j)
    assert delop & Vector.zero is S.Zero == divergence(Vector.zero)
    assert (delop & Vector.zero).doit() is S.Zero
    assert delop.dot(Vector.zero) == delop & Vector.zero
    assert (delop & i).doit() is S.Zero
    assert (delop & x ** 2 * i).doit() == 2 * x == divergence(x ** 2 * i)
    assert delop.dot(v, doit=True) == x * y + y * z + z * x == divergence(v)
    assert delop & v == delop.dot(v)
    assert delop.dot(1 / (x * y * z) * (i + j + k), doit=True) == -1 / (x * y * z ** 2) - 1 / (x * y ** 2 * z) - 1 / (x ** 2 * y * z)
    v = x * i + y * j + z * k
    assert delop & v == Derivative(C.x, C.x) + Derivative(C.y, C.y) + Derivative(C.z, C.z)
    assert delop.dot(v, doit=True) == 3 == divergence(v)
    assert delop & v == delop.dot(v)
    assert simplify((delop & v).doit()) == 3
    assert delop.gradient(0, doit=True) == Vector.zero == gradient(0)
    assert delop.gradient(0) == delop(0)
    assert delop(S.Zero).doit() == Vector.zero
    assert delop(x) == Derivative(C.x, C.x) * C.i + Derivative(C.x, C.y) * C.j + Derivative(C.x, C.z) * C.k
    assert delop(x).doit() == i == gradient(x)
    assert delop(x * y * z) == Derivative(C.x * C.y * C.z, C.x) * C.i + Derivative(C.x * C.y * C.z, C.y) * C.j + Derivative(C.x * C.y * C.z, C.z) * C.k
    assert delop.gradient(x * y * z, doit=True) == y * z * i + z * x * j + x * y * k == gradient(x * y * z)
    assert delop(x * y * z) == delop.gradient(x * y * z)
    assert delop(2 * x ** 2).doit() == 4 * x * i
    assert delop(a * sin(y) / x).doit() == -a * sin(y) / x ** 2 * i + a * cos(y) / x * j
    assert (Vector.zero & delop)(a) is S.Zero
    assert (Vector.zero & delop)(a).doit() is S.Zero
    assert (v & delop)(Vector.zero).doit() == Vector.zero
    assert (v & delop)(S.Zero).doit() is S.Zero
    assert (i & delop)(x).doit() == 1
    assert (j & delop)(y).doit() == 1
    assert (k & delop)(z).doit() == 1
    assert (i & delop)(x * y * z).doit() == y * z
    assert (v & delop)(x).doit() == x
    assert (v & delop)(x * y * z).doit() == 3 * x * y * z
    assert (v & delop)(x + y + z) == C.x + C.y + C.z
    assert (v & delop)(x + y + z).doit() == x + y + z
    assert (v & delop)(v).doit() == v
    assert (i & delop)(v).doit() == i
    assert (j & delop)(v).doit() == j
    assert (k & delop)(v).doit() == k
    assert (v & delop)(Vector.zero).doit() == Vector.zero
    assert laplacian(x * y * z) is S.Zero
    assert laplacian(x ** 2) == S(2)
    assert laplacian(x ** 2 * y ** 2 * z ** 2) == 2 * y ** 2 * z ** 2 + 2 * x ** 2 * z ** 2 + 2 * x ** 2 * y ** 2
    A = CoordSys3D('A', transformation='spherical', variable_names=['r', 'theta', 'phi'])
    B = CoordSys3D('B', transformation='cylindrical', variable_names=['r', 'theta', 'z'])
    assert laplacian(A.r + A.theta + A.phi) == 2 / A.r + cos(A.theta) / (A.r ** 2 * sin(A.theta))
    assert laplacian(B.r + B.theta + B.z) == 1 / B.r
    assert laplacian(x * y * z * (i + j + k)) == Vector.zero
    assert laplacian(x * y ** 2 * z * (i + j + k)) == 2 * x * z * i + 2 * x * z * j + 2 * x * z * k