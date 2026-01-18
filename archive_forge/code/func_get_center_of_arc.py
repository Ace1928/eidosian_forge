from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def get_center_of_arc(p1, p2, radius):
    """

    Args:
        p1:
        p2:
        radius:
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dd = np.sqrt(dx * dx + dy * dy)
    radical = np.power(radius / dd, 2) - 0.25
    if radical < 0:
        raise ValueError('Impossible to find center of arc because the arc is ill-defined')
    tt = np.sqrt(radical)
    if radius > 0:
        tt = -tt
    return ((p1[0] + p2[0]) / 2 - tt * dy, (p1[1] + p2[1]) / 2 + tt * dx)