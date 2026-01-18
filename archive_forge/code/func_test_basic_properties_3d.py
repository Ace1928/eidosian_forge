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
def test_basic_properties_3d():
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(x1, x1, x1)
    p5 = Point3D(x1, 1 + x1, 1)
    l1 = Line3D(p1, p2)
    l3 = Line3D(p3, p5)
    r1 = Ray3D(p1, Point3D(-1, 5, 0))
    r3 = Ray3D(p1, p2)
    s1 = Segment3D(p1, p2)
    assert Line3D((1, 1, 1), direction_ratio=[2, 3, 4]) == Line3D(Point3D(1, 1, 1), Point3D(3, 4, 5))
    assert Line3D((1, 1, 1), direction_ratio=[1, 5, 7]) == Line3D(Point3D(1, 1, 1), Point3D(2, 6, 8))
    assert Line3D((1, 1, 1), direction_ratio=[1, 2, 3]) == Line3D(Point3D(1, 1, 1), Point3D(2, 3, 4))
    assert Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)).direction_cosine == [1, 0, 0]
    assert Line3D(Line3D(p1, Point3D(0, 1, 0))) == Line3D(p1, Point3D(0, 1, 0))
    assert Ray3D(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))) == Ray3D(p1, Point3D(1, 0, 0))
    assert Line3D(p1, p2) != Line3D(p2, p1)
    assert l1 != l3
    assert l1 != Line3D(p3, Point3D(y1, y1, y1))
    assert r3 != r1
    assert Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1)) in Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2))
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)) in Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).xdirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).ydirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 2)).zdirection == S.Infinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(-2, 2, 2)).xdirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, -2, 2)).ydirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, -2)).zdirection == S.NegativeInfinity
    assert Ray3D(Point3D(0, 0, 0), Point3D(0, 2, 2)).xdirection == S.Zero
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 0, 2)).ydirection == S.Zero
    assert Ray3D(Point3D(0, 0, 0), Point3D(2, 2, 0)).zdirection == S.Zero
    assert p1 in l1
    assert p1 not in l3
    assert l1.direction_ratio == [1, 1, 1]
    assert s1.midpoint == Point3D(S.Half, S.Half, S.Half)
    assert Ray3D(p1, Point3D(0, 0, -1)).zdirection is S.NegativeInfinity