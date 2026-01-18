from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def powern_decreasing(xx, edges=None, nn=2):
    """

    Args:
        xx:
        edges:
        nn:
    """
    if edges is None:
        aa = 1.0 / np.power(-1.0, nn)
        return aa * np.power(xx - 1.0, nn)
    xx_scaled_and_clamped = scale_and_clamp(xx, edges[0], edges[1], 0.0, 1.0)
    return powern_decreasing(xx_scaled_and_clamped, nn=nn)