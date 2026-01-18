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
def test_intersection_2d():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p3 = Point(x1, x1)
    p4 = Point(y1, y1)
    l1 = Line(p1, p2)
    l3 = Line(Point(0, 0), Point(3, 4))
    r1 = Ray(Point(1, 1), Point(2, 2))
    r2 = Ray(Point(0, 0), Point(3, 4))
    r4 = Ray(p1, p2)
    r6 = Ray(Point(0, 1), Point(1, 2))
    r7 = Ray(Point(0.5, 0.5), Point(1, 1))
    s1 = Segment(p1, p2)
    s2 = Segment(Point(0.25, 0.25), Point(0.5, 0.5))
    s3 = Segment(Point(0, 0), Point(3, 4))
    assert intersection(l1, p1) == [p1]
    assert intersection(l1, Point(x1, 1 + x1)) == []
    assert intersection(l1, Line(p3, p4)) in [[l1], [Line(p3, p4)]]
    assert intersection(l1, l1.parallel_line(Point(x1, 1 + x1))) == []
    assert intersection(l3, l3) == [l3]
    assert intersection(l3, r2) == [r2]
    assert intersection(l3, s3) == [s3]
    assert intersection(s3, l3) == [s3]
    assert intersection(Segment(Point(-10, 10), Point(10, 10)), Segment(Point(-5, -5), Point(-5, 5))) == []
    assert intersection(r2, l3) == [r2]
    assert intersection(r1, Ray(Point(2, 2), Point(0, 0))) == [Segment(Point(1, 1), Point(2, 2))]
    assert intersection(r1, Ray(Point(1, 1), Point(-1, -1))) == [Point(1, 1)]
    assert intersection(r1, Segment(Point(0, 0), Point(2, 2))) == [Segment(Point(1, 1), Point(2, 2))]
    assert r4.intersection(s2) == [s2]
    assert r4.intersection(Segment(Point(2, 3), Point(3, 4))) == []
    assert r4.intersection(Segment(Point(-1, -1), Point(0.5, 0.5))) == [Segment(p1, Point(0.5, 0.5))]
    assert r4.intersection(Ray(p2, p1)) == [s1]
    assert Ray(p2, p1).intersection(r6) == []
    assert r4.intersection(r7) == r7.intersection(r4) == [r7]
    assert Ray3D((0, 0), (3, 0)).intersection(Ray3D((1, 0), (3, 0))) == [Ray3D((1, 0), (3, 0))]
    assert Ray3D((1, 0), (3, 0)).intersection(Ray3D((0, 0), (3, 0))) == [Ray3D((1, 0), (3, 0))]
    assert Ray(Point(0, 0), Point(0, 4)).intersection(Ray(Point(0, 1), Point(0, -1))) == [Segment(Point(0, 0), Point(0, 1))]
    assert Segment3D((0, 0), (3, 0)).intersection(Segment3D((1, 0), (2, 0))) == [Segment3D((1, 0), (2, 0))]
    assert Segment3D((1, 0), (2, 0)).intersection(Segment3D((0, 0), (3, 0))) == [Segment3D((1, 0), (2, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(Segment3D((3, 0), (4, 0))) == [Point3D((3, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(Segment3D((2, 0), (5, 0))) == [Segment3D((2, 0), (3, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(Segment3D((-2, 0), (1, 0))) == [Segment3D((0, 0), (1, 0))]
    assert Segment3D((0, 0), (3, 0)).intersection(Segment3D((-2, 0), (0, 0))) == [Point3D(0, 0)]
    assert s1.intersection(Segment(Point(1, 1), Point(2, 2))) == [Point(1, 1)]
    assert s1.intersection(Segment(Point(0.5, 0.5), Point(1.5, 1.5))) == [Segment(Point(0.5, 0.5), p2)]
    assert s1.intersection(Segment(Point(4, 4), Point(5, 5))) == []
    assert s1.intersection(Segment(Point(-1, -1), p1)) == [p1]
    assert s1.intersection(Segment(Point(-1, -1), Point(0.5, 0.5))) == [Segment(p1, Point(0.5, 0.5))]
    assert s1.intersection(Line(Point(1, 0), Point(2, 1))) == []
    assert s1.intersection(s2) == [s2]
    assert s2.intersection(s1) == [s2]
    assert asa(120, 8, 52) == Triangle(Point(0, 0), Point(8, 0), Point(-4 * cos(19 * pi / 90) / sin(2 * pi / 45), 4 * sqrt(3) * cos(19 * pi / 90) / sin(2 * pi / 45)))
    assert Line((0, 0), (1, 1)).intersection(Ray((1, 0), (1, 2))) == [Point(1, 1)]
    assert Line((0, 0), (1, 1)).intersection(Segment((1, 0), (1, 2))) == [Point(1, 1)]
    assert Ray((0, 0), (1, 1)).intersection(Ray((1, 0), (1, 2))) == [Point(1, 1)]
    assert Ray((0, 0), (1, 1)).intersection(Segment((1, 0), (1, 2))) == [Point(1, 1)]
    assert Ray((0, 0), (10, 10)).contains(Segment((1, 1), (2, 2))) is True
    assert Segment((1, 1), (2, 2)) in Line((0, 0), (10, 10))
    assert s1.intersection(Ray((1, 1), (4, 4))) == [Point(1, 1)]