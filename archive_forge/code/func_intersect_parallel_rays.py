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
def intersect_parallel_rays(ray1, ray2):
    if ray1.direction.dot(ray2.direction) > 0:
        return [ray2] if ray1._span_test(ray2.p1) >= 0 else [ray1]
    else:
        st = ray1._span_test(ray2.p1)
        if st < 0:
            return []
        elif st == 0:
            return [ray2.p1]
        return [Segment(ray1.p1, ray2.p1)]