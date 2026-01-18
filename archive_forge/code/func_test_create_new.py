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
def test_create_new():
    a = CoordSys3D('a')
    c = a.create_new('c', transformation='spherical')
    assert c._parent == a
    assert c.transformation_to_parent() == (c.r * sin(c.theta) * cos(c.phi), c.r * sin(c.theta) * sin(c.phi), c.r * cos(c.theta))
    assert c.transformation_from_parent() == (sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2), acos(a.z / sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)), atan2(a.y, a.x))