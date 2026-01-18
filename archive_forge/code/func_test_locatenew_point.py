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
def test_locatenew_point():
    """
    Tests Point class, and locate_new method in CoordSys3D.
    """
    A = CoordSys3D('A')
    assert isinstance(A.origin, Point)
    v = a * A.i + b * A.j + c * A.k
    C = A.locate_new('C', v)
    assert C.origin.position_wrt(A) == C.position_wrt(A) == C.origin.position_wrt(A.origin) == v
    assert A.origin.position_wrt(C) == A.position_wrt(C) == A.origin.position_wrt(C.origin) == -v
    assert A.origin.express_coordinates(C) == (-a, -b, -c)
    p = A.origin.locate_new('p', -v)
    assert p.express_coordinates(A) == (-a, -b, -c)
    assert p.position_wrt(C.origin) == p.position_wrt(C) == -2 * v
    p1 = p.locate_new('p1', 2 * v)
    assert p1.position_wrt(C.origin) == Vector.zero
    assert p1.express_coordinates(C) == (0, 0, 0)
    p2 = p.locate_new('p2', A.i)
    assert p1.position_wrt(p2) == 2 * v - A.i
    assert p2.express_coordinates(C) == (-2 * a + 1, -2 * b, -2 * c)