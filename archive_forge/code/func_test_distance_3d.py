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
def test_distance_3d():
    p1, p2 = (Point3D(0, 0, 0), Point3D(1, 1, 1))
    p3 = Point3D(Rational(3) / 2, Rational(3) / 2, Rational(3) / 2)
    s1 = Segment3D(Point3D(0, 0, 0), Point3D(1, 1, 1))
    s2 = Segment3D(Point3D(S.Half, S.Half, S.Half), Point3D(1, 0, 1))
    r = Ray3D(p1, p2)
    assert s1.distance(p1) == 0
    assert s2.distance(p1) == sqrt(3) / 2
    assert s2.distance(p3) == 2 * sqrt(6) / 3
    assert s1.distance((0, 0, 0)) == 0
    assert s2.distance((0, 0, 0)) == sqrt(3) / 2
    assert s1.distance(p1) == 0
    assert s2.distance(p1) == sqrt(3) / 2
    assert s2.distance(p3) == 2 * sqrt(6) / 3
    assert s1.distance((0, 0, 0)) == 0
    assert s2.distance((0, 0, 0)) == sqrt(3) / 2
    assert Line3D(p1, p2).distance(Point3D(-1, 1, 1)) == 2 * sqrt(6) / 3
    assert Line3D(p1, p2).distance(Point3D(1, -1, 1)) == 2 * sqrt(6) / 3
    assert Line3D(p1, p2).distance(Point3D(2, 2, 2)) == 0
    assert Line3D(p1, p2).distance((2, 2, 2)) == 0
    assert Line3D(p1, p2).distance((1, -1, 1)) == 2 * sqrt(6) / 3
    assert Line3D((0, 0, 0), (0, 1, 0)).distance(p1) == 0
    assert Line3D((0, 0, 0), (0, 1, 0)).distance(p2) == sqrt(2)
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(p1) == 0
    assert Line3D((0, 0, 0), (1, 0, 0)).distance(p2) == sqrt(2)
    assert r.distance(Point3D(-1, -1, -1)) == sqrt(3)
    assert r.distance(Point3D(1, 1, 1)) == 0
    assert r.distance((-1, -1, -1)) == sqrt(3)
    assert r.distance((1, 1, 1)) == 0
    assert Ray3D((0, 0, 0), (1, 1, 2)).distance((-1, -1, 2)) == 4 * sqrt(3) / 3
    assert Ray3D((1, 1, 1), (2, 2, 2)).distance(Point3D(1.5, -3, -1)) == Rational(9) / 2
    assert Ray3D((1, 1, 1), (2, 2, 2)).distance(Point3D(1.5, 3, 1)) == sqrt(78) / 6