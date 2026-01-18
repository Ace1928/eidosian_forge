from sympy.core import expand
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sec
from sympy.geometry.line import Segment2D
from sympy.geometry.point import Point2D
from sympy.geometry import (Circle, Ellipse, GeometryError, Line, Point,
from sympy.testing.pytest import raises, slow
from sympy.integrals.integrals import integrate
from sympy.functions.special.elliptic_integrals import elliptic_e
from sympy.functions.elementary.miscellaneous import Max
def test_director_circle():
    x, y, a, b = symbols('x y a b')
    e = Ellipse((x, y), a, b)
    assert e.director_circle() == Circle((x, y), sqrt(a ** 2 + b ** 2))
    assert Circle((3, 4), 8).director_circle() == Circle((3, 4), 8 * sqrt(2))