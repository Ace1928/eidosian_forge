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
def test_lame_coefficients():
    a = CoordSys3D('a', 'spherical')
    assert a.lame_coefficients() == (1, a.r, sin(a.theta) * a.r)
    a = CoordSys3D('a')
    assert a.lame_coefficients() == (1, 1, 1)
    a = CoordSys3D('a', 'cartesian')
    assert a.lame_coefficients() == (1, 1, 1)
    a = CoordSys3D('a', 'cylindrical')
    assert a.lame_coefficients() == (1, a.r, 1)