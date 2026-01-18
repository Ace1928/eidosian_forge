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
def are_similar(e1, e2):
    """Are two geometrical entities similar.

    Can one geometrical entity be uniformly scaled to the other?

    Parameters
    ==========

    e1 : GeometryEntity
    e2 : GeometryEntity

    Returns
    =======

    are_similar : boolean

    Raises
    ======

    GeometryError
        When `e1` and `e2` cannot be compared.

    Notes
    =====

    If the two objects are equal then they are similar.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.is_similar

    Examples
    ========

    >>> from sympy import Point, Circle, Triangle, are_similar
    >>> c1, c2 = Circle(Point(0, 0), 4), Circle(Point(1, 4), 3)
    >>> t1 = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    >>> t2 = Triangle(Point(0, 0), Point(2, 0), Point(0, 2))
    >>> t3 = Triangle(Point(0, 0), Point(3, 0), Point(0, 1))
    >>> are_similar(t1, t2)
    True
    >>> are_similar(t1, t3)
    False

    """
    if e1 == e2:
        return True
    is_similar1 = getattr(e1, 'is_similar', None)
    if is_similar1:
        return is_similar1(e2)
    is_similar2 = getattr(e2, 'is_similar', None)
    if is_similar2:
        return is_similar2(e1)
    n1 = e1.__class__.__name__
    n2 = e2.__class__.__name__
    raise GeometryError('Cannot test similarity between %s and %s' % (n1, n2))