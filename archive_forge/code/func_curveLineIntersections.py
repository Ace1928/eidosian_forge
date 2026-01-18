from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def curveLineIntersections(curve, line):
    """Finds intersections between a curve and a line.

    Args:
        curve: List of coordinates of the curve segment as 2D tuples.
        line: List of coordinates of the line segment as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::
        >>> curve = [ (100, 240), (30, 60), (210, 230), (160, 30) ]
        >>> line  = [ (25, 260), (230, 20) ]
        >>> intersections = curveLineIntersections(curve, line)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (84.9000930760723, 189.87306176459828)
    """
    if len(curve) == 3:
        pointFinder = quadraticPointAtT
    elif len(curve) == 4:
        pointFinder = cubicPointAtT
    else:
        raise ValueError('Unknown curve degree')
    intersections = []
    for t in _curve_line_intersections_t(curve, line):
        pt = pointFinder(*curve, t)
        line_t = _line_t_of_pt(*line, pt)
        pt = linePointAtT(*line, line_t)
        intersections.append(Intersection(pt=pt, t1=t, t2=line_t))
    return intersections