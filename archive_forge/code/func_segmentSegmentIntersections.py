from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def segmentSegmentIntersections(seg1, seg2):
    """Finds intersections between two segments.

    Args:
        seg1: List of coordinates of the first segment as 2D tuples.
        seg2: List of coordinates of the second segment as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::
        >>> curve1 = [ (10,100), (90,30), (40,140), (220,220) ]
        >>> curve2 = [ (5,150), (180,20), (80,250), (210,190) ]
        >>> intersections = segmentSegmentIntersections(curve1, curve2)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (81.7831487395506, 109.88904552375288)
        >>> curve3 = [ (100, 240), (30, 60), (210, 230), (160, 30) ]
        >>> line  = [ (25, 260), (230, 20) ]
        >>> intersections = segmentSegmentIntersections(curve3, line)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (84.9000930760723, 189.87306176459828)

    """
    swapped = False
    if len(seg2) > len(seg1):
        seg2, seg1 = (seg1, seg2)
        swapped = True
    if len(seg1) > 2:
        if len(seg2) > 2:
            intersections = curveCurveIntersections(seg1, seg2)
        else:
            intersections = curveLineIntersections(seg1, seg2)
    elif len(seg1) == 2 and len(seg2) == 2:
        intersections = lineLineIntersections(*seg1, *seg2)
    else:
        raise ValueError("Couldn't work out which intersection function to use")
    if not swapped:
        return intersections
    return [Intersection(pt=i.pt, t1=i.t2, t2=i.t1) for i in intersections]