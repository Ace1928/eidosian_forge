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
def test_issue_8615():
    a = Line3D(Point3D(6, 5, 0), Point3D(6, -6, 0))
    b = Line3D(Point3D(6, -1, 19 / 10), Point3D(6, -1, 0))
    assert a.intersection(b) == [Point3D(6, -1, 0)]