from sympy.core.basic import Basic
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.parameters import evaluate
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Line, Point, Point2D, Point3D, Line3D, Plane
from sympy.geometry.entity import rotate, scale, translate, GeometryEntity
from sympy.matrices import Matrix
from sympy.utilities.iterables import subsets, permutations, cartes
from sympy.utilities.misc import Undecidable
from sympy.testing.pytest import raises, warns
def test_Point2D():
    p1 = Point2D(1, 5)
    p2 = Point2D(4, 2.5)
    p3 = (6, 3)
    assert p1.distance(p2) == sqrt(61) / 2
    assert p2.distance(p3) == sqrt(17) / 2
    assert p1.x == 1
    assert p1.y == 5
    assert p2.x == 4
    assert p2.y == S(5) / 2
    assert p1.coordinates == (1, 5)
    assert p2.coordinates == (4, S(5) / 2)
    assert p1.bounds == (1, 5, 1, 5)