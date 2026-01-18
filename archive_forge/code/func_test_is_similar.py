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
def test_is_similar():
    p1 = Point(2000, 2000)
    p2 = p1.scale(2, 2)
    r1 = Ray3D(Point3D(1, 1, 1), Point3D(1, 0, 0))
    r2 = Ray(Point(0, 0), Point(0, 1))
    s1 = Segment(Point(0, 0), p1)
    assert s1.is_similar(Segment(p1, p2))
    assert s1.is_similar(r2) is False
    assert r1.is_similar(Line3D(Point3D(1, 1, 1), Point3D(1, 0, 0))) is True
    assert r1.is_similar(Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))) is False