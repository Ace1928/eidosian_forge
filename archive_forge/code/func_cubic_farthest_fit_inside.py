from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (
@cython.cfunc
@cython.returns(cython.int)
@cython.locals(tolerance=cython.double, p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
@cython.locals(mid=cython.complex, deriv3=cython.complex)
def cubic_farthest_fit_inside(p0, p1, p2, p3, tolerance):
    """Check if a cubic Bezier lies within a given distance of the origin.

    "Origin" means *the* origin (0,0), not the start of the curve. Note that no
    checks are made on the start and end positions of the curve; this function
    only checks the inside of the curve.

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.
        tolerance (double): Distance from origin.

    Returns:
        bool: True if the cubic Bezier ``p`` entirely lies within a distance
        ``tolerance`` of the origin, False otherwise.
    """
    if abs(p2) <= tolerance and abs(p1) <= tolerance:
        return True
    mid = (p0 + 3 * (p1 + p2) + p3) * 0.125
    if abs(mid) > tolerance:
        return False
    deriv3 = (p3 + p2 - p1 - p0) * 0.125
    return cubic_farthest_fit_inside(p0, (p0 + p1) * 0.5, mid - deriv3, mid, tolerance) and cubic_farthest_fit_inside(mid, mid + deriv3, (p2 + p3) * 0.5, p3, tolerance)