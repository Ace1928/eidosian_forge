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
def test_issue_2941():

    def _check():
        for f, g in cartes(*[(Line, Ray, Segment)] * 2):
            l1 = f(a, b)
            l2 = g(c, d)
            assert l1.intersection(l2) == l2.intersection(l1)
    c, d = ((-2, -2), (-2, 0))
    a, b = ((0, 0), (1, 1))
    _check()
    c, d = ((-2, -3), (-2, 0))
    _check()