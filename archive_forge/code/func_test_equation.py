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
def test_equation():
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    l1 = Line(p1, p2)
    l3 = Line(Point(x1, x1), Point(x1, 1 + x1))
    assert simplify(l1.equation()) in (x - y, y - x)
    assert simplify(l3.equation()) in (x - x1, x1 - x)
    assert simplify(l1.equation()) in (x - y, y - x)
    assert simplify(l3.equation()) in (x - x1, x1 - x)
    assert Line(p1, Point(1, 0)).equation(x=x, y=y) == y
    assert Line(p1, Point(0, 1)).equation() == x
    assert Line(Point(2, 0), Point(2, 1)).equation() == x - 2
    assert Line(p2, Point(2, 1)).equation() == y - 1
    assert Line3D(Point(x1, x1, x1), Point(y1, y1, y1)).equation() == (-x + y, -x + z)
    assert Line3D(Point(1, 2, 3), Point(2, 3, 4)).equation() == (-x + y - 1, -x + z - 2)
    assert Line3D(Point(1, 2, 3), Point(1, 3, 4)).equation() == (x - 1, -y + z - 1)
    assert Line3D(Point(1, 2, 3), Point(2, 2, 4)).equation() == (y - 2, -x + z - 2)
    assert Line3D(Point(1, 2, 3), Point(2, 3, 3)).equation() == (-x + y - 1, z - 3)
    assert Line3D(Point(1, 2, 3), Point(1, 2, 4)).equation() == (x - 1, y - 2)
    assert Line3D(Point(1, 2, 3), Point(1, 3, 3)).equation() == (x - 1, z - 3)
    assert Line3D(Point(1, 2, 3), Point(2, 2, 3)).equation() == (y - 2, z - 3)