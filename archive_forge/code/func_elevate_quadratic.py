from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (
@cython.locals(p0=cython.complex, p1=cython.complex, p2=cython.complex, p1_2_3=cython.complex)
def elevate_quadratic(p0, p1, p2):
    """Given a quadratic bezier curve, return its degree-elevated cubic."""
    p1_2_3 = p1 * (2 / 3)
    return (p0, p0 * (1 / 3) + p1_2_3, p2 * (1 / 3) + p1_2_3, p2)