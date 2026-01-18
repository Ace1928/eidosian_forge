from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, Ellipse, GeometryError, Point, Point2D,
from sympy.testing.pytest import raises, slow, warns
from sympy.core.random import verify_numerically
from sympy.geometry.polygon import rad, deg
from sympy.integrals.integrals import integrate
def test_issue_12966():
    poly = Polygon(Point(0, 0), Point(0, 10), Point(5, 10), Point(5, 5), Point(10, 5), Point(10, 0))
    t = Symbol('t')
    pt = poly.arbitrary_point(t)
    DELTA = 5 / poly.perimeter
    assert [pt.subs(t, DELTA * i) for i in range(int(1 / DELTA))] == [Point(0, 0), Point(0, 5), Point(0, 10), Point(5, 10), Point(5, 5), Point(10, 5), Point(10, 0), Point(5, 0)]