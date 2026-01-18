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
def test_intersection_3d():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    l1 = Line3D(p1, p2)
    l2 = Line3D(Point3D(0, 0, 0), Point3D(3, 4, 0))
    r1 = Ray3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
    r2 = Ray3D(Point3D(0, 0, 0), Point3D(3, 4, 0))
    s1 = Segment3D(Point3D(0, 0, 0), Point3D(3, 4, 0))
    assert intersection(l1, p1) == [p1]
    assert intersection(l1, Point3D(x1, 1 + x1, 1)) == []
    assert intersection(l1, l1.parallel_line(p1)) == [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 1))]
    assert intersection(l2, r2) == [r2]
    assert intersection(l2, s1) == [s1]
    assert intersection(r2, l2) == [r2]
    assert intersection(r1, Ray3D(Point3D(1, 1, 1), Point3D(-1, -1, -1))) == [Point3D(1, 1, 1)]
    assert intersection(r1, Segment3D(Point3D(0, 0, 0), Point3D(2, 2, 2))) == [Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))]
    assert intersection(Ray3D(Point3D(1, 0, 0), Point3D(-1, 0, 0)), Ray3D(Point3D(0, 1, 0), Point3D(0, -1, 0))) == [Point3D(0, 0, 0)]
    assert intersection(r1, Ray3D(Point3D(2, 2, 2), Point3D(0, 0, 0))) == [Segment3D(Point3D(1, 1, 1), Point3D(2, 2, 2))]
    assert intersection(s1, r2) == [s1]
    assert Line3D(Point3D(4, 0, 1), Point3D(0, 4, 1)).intersection(Line3D(Point3D(0, 0, 1), Point3D(4, 4, 1))) == [Point3D(2, 2, 1)]
    assert Line3D((0, 1, 2), (0, 2, 3)).intersection(Line3D((0, 1, 2), (0, 1, 1))) == [Point3D(0, 1, 2)]
    assert Line3D((0, 0), (t, t)).intersection(Line3D((0, 1), (t, t))) == [Point3D(t, t)]
    assert Ray3D(Point3D(0, 0, 0), Point3D(0, 4, 0)).intersection(Ray3D(Point3D(0, 1, 1), Point3D(0, -1, 1))) == []