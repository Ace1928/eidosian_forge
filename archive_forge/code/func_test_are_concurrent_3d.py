from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.sets import EmptySet
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, GeometryError, Line, Point, Ray,
from sympy.geometry.line import Undecidable
from sympy.geometry.polygon import _asa as asa
from sympy.utilities.iterables import cartes
from sympy.testing.pytest import raises, warns
def test_are_concurrent_3d():
    p1 = Point3D(0, 0, 0)
    l1 = Line(p1, Point3D(1, 1, 1))
    parallel_1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
    parallel_2 = Line3D(Point3D(0, 1, 0), Point3D(1, 1, 0))
    assert Line3D.are_concurrent(l1) is False
    assert Line3D.are_concurrent(l1, Line(Point3D(x1, x1, x1), Point3D(y1, y1, y1))) is False
    assert Line3D.are_concurrent(l1, Line3D(p1, Point3D(x1, x1, x1)), Line(Point3D(x1, x1, x1), Point3D(x1, 1 + x1, 1))) is True
    assert Line3D.are_concurrent(parallel_1, parallel_2) is False