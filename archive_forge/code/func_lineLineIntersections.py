from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def lineLineIntersections(s1, e1, s2, e2):
    """Finds intersections between two line segments.

    Args:
        s1, e1: Coordinates of the first line as 2D tuples.
        s2, e2: Coordinates of the second line as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::

        >>> a = lineLineIntersections( (310,389), (453, 222), (289, 251), (447, 367))
        >>> len(a)
        1
        >>> intersection = a[0]
        >>> intersection.pt
        (374.44882952482897, 313.73458370177315)
        >>> (intersection.t1, intersection.t2)
        (0.45069111555824465, 0.5408153767394238)
    """
    s1x, s1y = s1
    e1x, e1y = e1
    s2x, s2y = s2
    e2x, e2y = e2
    if math.isclose(s2x, e2x) and math.isclose(s1x, e1x) and (not math.isclose(s1x, s2x)):
        return []
    if math.isclose(s2y, e2y) and math.isclose(s1y, e1y) and (not math.isclose(s1y, s2y)):
        return []
    if math.isclose(s2x, e2x) and math.isclose(s2y, e2y):
        return []
    if math.isclose(s1x, e1x) and math.isclose(s1y, e1y):
        return []
    if math.isclose(e1x, s1x):
        x = s1x
        slope34 = (e2y - s2y) / (e2x - s2x)
        y = slope34 * (x - s2x) + s2y
        pt = (x, y)
        return [Intersection(pt=pt, t1=_line_t_of_pt(s1, e1, pt), t2=_line_t_of_pt(s2, e2, pt))]
    if math.isclose(s2x, e2x):
        x = s2x
        slope12 = (e1y - s1y) / (e1x - s1x)
        y = slope12 * (x - s1x) + s1y
        pt = (x, y)
        return [Intersection(pt=pt, t1=_line_t_of_pt(s1, e1, pt), t2=_line_t_of_pt(s2, e2, pt))]
    slope12 = (e1y - s1y) / (e1x - s1x)
    slope34 = (e2y - s2y) / (e2x - s2x)
    if math.isclose(slope12, slope34):
        return []
    x = (slope12 * s1x - s1y - slope34 * s2x + s2y) / (slope12 - slope34)
    y = slope12 * (x - s1x) + s1y
    pt = (x, y)
    if _both_points_are_on_same_side_of_origin(pt, e1, s1) and _both_points_are_on_same_side_of_origin(pt, s2, e2):
        return [Intersection(pt=pt, t1=_line_t_of_pt(s1, e1, pt), t2=_line_t_of_pt(s2, e2, pt))]
    return []