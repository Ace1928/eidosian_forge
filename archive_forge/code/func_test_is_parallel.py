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
def test_is_parallel():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(x1, x1, x1)
    l2 = Line(Point(x1, x1), Point(y1, y1))
    l2_1 = Line(Point(x1, x1), Point(x1, 1 + x1))
    assert Line.is_parallel(Line(Point(0, 0), Point(1, 1)), l2)
    assert Line.is_parallel(l2, Line(Point(x1, x1), Point(x1, 1 + x1))) is False
    assert Line.is_parallel(l2, l2.parallel_line(Point(-x1, x1)))
    assert Line.is_parallel(l2_1, l2_1.parallel_line(Point(0, 0)))
    assert Line3D(p1, p2).is_parallel(Line3D(p1, p2))
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).is_parallel(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) is False
    assert Line3D(p1, p2).parallel_line(p3) == Line3D(Point3D(x1, x1, x1), Point3D(x1 + 1, x1 + 1, x1 + 1))
    assert Line3D(p1, p2).parallel_line(p3.args) == Line3D(Point3D(x1, x1, x1), Point3D(x1 + 1, x1 + 1, x1 + 1))
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).is_parallel(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) is False