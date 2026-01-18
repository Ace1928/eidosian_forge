from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.returns(cython.complex)
@cython.locals(t=cython.double, pt1=cython.complex, pt2=cython.complex, pt3=cython.complex, pt4=cython.complex)
@cython.locals(t2=cython.double, _1_t=cython.double, _1_t_2=cython.double)
def cubicPointAtTC(pt1, pt2, pt3, pt4, t):
    """Finds the point at time `t` on a cubic curve.

    Args:
        pt1, pt2, pt3, pt4: Coordinates of the curve as complex numbers.
        t: The time along the curve.

    Returns:
        A complex number with the coordinates of the point.
    """
    t2 = t * t
    _1_t = 1 - t
    _1_t_2 = _1_t * _1_t
    return _1_t_2 * _1_t * pt1 + 3 * (_1_t_2 * t * pt2 + _1_t * t2 * pt3) + t2 * t * pt4