from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.returns(cython.double)
@cython.locals(pt1=cython.complex, pt2=cython.complex, pt3=cython.complex)
@cython.locals(v0=cython.double, v1=cython.double, v2=cython.double)
def approximateQuadraticArcLengthC(pt1, pt2, pt3):
    """Calculates the arc length for a quadratic Bezier segment.

    Uses Gauss-Legendre quadrature for a branch-free approximation.
    See :func:`calcQuadraticArcLength` for a slower but more accurate result.

    Args:
        pt1: Start point of the Bezier as a complex number.
        pt2: Handle point of the Bezier as a complex number.
        pt3: End point of the Bezier as a complex number.

    Returns:
        Approximate arc length value.
    """
    v0 = abs(-0.492943519233745 * pt1 + 0.430331482911935 * pt2 + 0.0626120363218102 * pt3)
    v1 = abs(pt3 - pt1) * 0.4444444444444444
    v2 = abs(-0.0626120363218102 * pt1 - 0.430331482911935 * pt2 + 0.492943519233745 * pt3)
    return v0 + v1 + v2