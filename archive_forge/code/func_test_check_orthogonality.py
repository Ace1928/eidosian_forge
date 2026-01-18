from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.scalar import BaseScalar
from sympy.core.function import expand
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector.functions import express
from sympy.vector.point import Point
from sympy.vector.vector import Vector
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
def test_check_orthogonality():
    x, y, z = symbols('x y z')
    u, v = symbols('u, v')
    a = CoordSys3D('a', transformation=((x, y, z), (x * sin(y) * cos(z), x * sin(y) * sin(z), x * cos(y))))
    assert a._check_orthogonality(a._transformation) is True
    a = CoordSys3D('a', transformation=((x, y, z), (x * cos(y), x * sin(y), z)))
    assert a._check_orthogonality(a._transformation) is True
    a = CoordSys3D('a', transformation=((u, v, z), (cosh(u) * cos(v), sinh(u) * sin(v), z)))
    assert a._check_orthogonality(a._transformation) is True
    raises(ValueError, lambda: CoordSys3D('a', transformation=((x, y, z), (x, x, z))))
    raises(ValueError, lambda: CoordSys3D('a', transformation=((x, y, z), (x * sin(y / 2) * cos(z), x * sin(y) * sin(z), x * cos(y)))))