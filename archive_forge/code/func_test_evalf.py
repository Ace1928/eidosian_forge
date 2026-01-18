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
def test_evalf():
    A = CoordSys3D('A')
    v = 3 * A.i + 4 * A.j + a * A.k
    assert v.n() == v.evalf()
    assert v.evalf(subs={a: 1}) == v.subs(a, 1).evalf()