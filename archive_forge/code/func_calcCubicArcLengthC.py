from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.returns(cython.double)
@cython.locals(pt1=cython.complex, pt2=cython.complex, pt3=cython.complex, pt4=cython.complex)
@cython.locals(tolerance=cython.double, mult=cython.double)
def calcCubicArcLengthC(pt1, pt2, pt3, pt4, tolerance=0.005):
    """Calculates the arc length for a cubic Bezier segment.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers.
        tolerance: Controls the precision of the calcuation.

    Returns:
        Arc length value.
    """
    mult = 1.0 + 1.5 * tolerance
    return _calcCubicArcLengthCRecurse(mult, pt1, pt2, pt3, pt4)