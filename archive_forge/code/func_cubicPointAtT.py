from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def cubicPointAtT(pt1, pt2, pt3, pt4, t):
    """Finds the point at time `t` on a cubic curve.

    Args:
        pt1, pt2, pt3, pt4: Coordinates of the curve as 2D tuples.
        t: The time along the curve.

    Returns:
        A 2D tuple with the coordinates of the point.
    """
    t2 = t * t
    _1_t = 1 - t
    _1_t_2 = _1_t * _1_t
    x = _1_t_2 * _1_t * pt1[0] + 3 * (_1_t_2 * t * pt2[0] + _1_t * t2 * pt3[0]) + t2 * t * pt4[0]
    y = _1_t_2 * _1_t * pt1[1] + 3 * (_1_t_2 * t * pt2[1] + _1_t * t2 * pt3[1]) + t2 * t * pt4[1]
    return (x, y)