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
def lines_close(l1, l2, prec):
    """ tests whether l1 and 12 are within 10**(-prec)
        of each other """
    return abs(l1.p1 - l2.p1) < 10 ** (-prec) and abs(l1.p2 - l2.p2) < 10 ** (-prec)