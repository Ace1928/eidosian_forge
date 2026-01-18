from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
def decomposeSuperBezierSegment(points):
    """Split the SuperBezier described by 'points' into a list of regular
    bezier segments. The 'points' argument must be a sequence with length
    3 or greater, containing (x, y) coordinates. The last point is the
    destination on-curve point, the rest of the points are off-curve points.
    The start point should not be supplied.

    This function returns a list of (pt1, pt2, pt3) tuples, which each
    specify a regular curveto-style bezier segment.
    """
    n = len(points) - 1
    assert n > 1
    bezierSegments = []
    pt1, pt2, pt3 = (points[0], None, None)
    for i in range(2, n + 1):
        nDivisions = min(i, 3, n - i + 2)
        for j in range(1, nDivisions):
            factor = j / nDivisions
            temp1 = points[i - 1]
            temp2 = points[i - 2]
            temp = (temp2[0] + factor * (temp1[0] - temp2[0]), temp2[1] + factor * (temp1[1] - temp2[1]))
            if pt2 is None:
                pt2 = temp
            else:
                pt3 = (0.5 * (pt2[0] + temp[0]), 0.5 * (pt2[1] + temp[1]))
                bezierSegments.append((pt1, pt2, pt3))
                pt1, pt2, pt3 = (temp, None, None)
    bezierSegments.append((pt1, points[-2], points[-1]))
    return bezierSegments