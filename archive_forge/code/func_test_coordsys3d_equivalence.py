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
def test_coordsys3d_equivalence():
    A = CoordSys3D('A')
    A1 = CoordSys3D('A')
    assert A1 == A
    B = CoordSys3D('B')
    assert A != B