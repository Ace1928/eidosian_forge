from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def approximateCubicArcLength(pt1, pt2, pt3, pt4):
    """Approximates the arc length for a cubic Bezier segment.

    Uses Gauss-Lobatto quadrature with n=5 points to approximate arc length.
    See :func:`calcCubicArcLength` for a slower but more accurate result.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as 2D tuples.

    Returns:
        Arc length value.

    Example::

        >>> approximateCubicArcLength((0, 0), (25, 100), (75, 100), (100, 0))
        190.04332968932817
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, 50), (100, 100))
        154.8852074945903
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, 0), (150, 0)) # line; exact result should be 150.
        149.99999999999991
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, 0), (-50, 0)) # cusp; exact result should be 150.
        136.9267662156362
        >>> approximateCubicArcLength((0, 0), (50, 0), (100, -50), (-50, 0)) # cusp
        154.80848416537057
    """
    return approximateCubicArcLengthC(complex(*pt1), complex(*pt2), complex(*pt3), complex(*pt4))