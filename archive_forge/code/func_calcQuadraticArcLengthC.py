from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.returns(cython.double)
@cython.locals(pt1=cython.complex, pt2=cython.complex, pt3=cython.complex, d0=cython.complex, d1=cython.complex, d=cython.complex, n=cython.complex)
@cython.locals(scale=cython.double, origDist=cython.double, a=cython.double, b=cython.double, x0=cython.double, x1=cython.double, Len=cython.double)
def calcQuadraticArcLengthC(pt1, pt2, pt3):
    """Calculates the arc length for a quadratic Bezier segment.

    Args:
        pt1: Start point of the Bezier as a complex number.
        pt2: Handle point of the Bezier as a complex number.
        pt3: End point of the Bezier as a complex number.

    Returns:
        Arc length value.
    """
    d0 = pt2 - pt1
    d1 = pt3 - pt2
    d = d1 - d0
    n = d * 1j
    scale = abs(n)
    if scale == 0.0:
        return abs(pt3 - pt1)
    origDist = _dot(n, d0)
    if abs(origDist) < epsilon:
        if _dot(d0, d1) >= 0:
            return abs(pt3 - pt1)
        a, b = (abs(d0), abs(d1))
        return (a * a + b * b) / (a + b)
    x0 = _dot(d, d0) / origDist
    x1 = _dot(d, d1) / origDist
    Len = abs(2 * (_intSecAtan(x1) - _intSecAtan(x0)) * origDist / (scale * (x1 - x0)))
    return Len