from sympy.core.containers import Tuple
from sympy.core.evalf import N
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, oo, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (_pi_coeff, acos, tan, atan2)
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .point import Point, Point3D
from .util import find, intersection
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.sets.sets import Intersection
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import Undecidable, filldedent
import random
def proj_point(p):
    return Point.project(p - self.p1, self.direction) + self.p1