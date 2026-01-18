from collections import deque
from math import sqrt as _sqrt
from .entity import GeometryEntity
from .exceptions import GeometryError
from .point import Point, Point2D, Point3D
from sympy.core.containers import OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, expand_mul
from sympy.core.sorting import ordered
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.polys.polytools import cancel
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import is_sequence
def rotatingCalipers(Points):
    U, L = convex_hull(*Points, **{'polygon': False})
    if L is None:
        if isinstance(U, Point):
            raise ValueError('At least two distinct points must be given.')
        yield U.args
    else:
        i = 0
        j = len(L) - 1
        while i < len(U) - 1 or j > 0:
            yield (U[i], L[j])
            if i == len(U) - 1:
                j -= 1
            elif j == 0:
                i += 1
            elif (U[i + 1].y - U[i].y) * (L[j].x - L[j - 1].x) > (L[j].y - L[j - 1].y) * (U[i + 1].x - U[i].x):
                i += 1
            else:
                j -= 1