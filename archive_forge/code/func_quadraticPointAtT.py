from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def quadraticPointAtT(pt1, pt2, pt3, t):
    """Finds the point at time `t` on a quadratic curve.

    Args:
        pt1, pt2, pt3: Coordinates of the curve as 2D tuples.
        t: The time along the curve.

    Returns:
        A 2D tuple with the coordinates of the point.
    """
    x = (1 - t) * (1 - t) * pt1[0] + 2 * (1 - t) * t * pt2[0] + t * t * pt3[0]
    y = (1 - t) * (1 - t) * pt1[1] + 2 * (1 - t) * t * pt2[1] + t * t * pt3[1]
    return (x, y)