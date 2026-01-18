from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def power2_inverse_powern_decreasing(xx, edges=None, prefactor=None, powern=2.0):
    """

    Args:
        xx:
        edges:
        prefactor:
        powern:
    """
    if edges is None:
        aa = 1.0 / np.power(-1.0, 2) if prefactor is None else prefactor
        return aa * np.power(xx - 1.0, 2) / xx ** powern
    xx_scaled_and_clamped = scale_and_clamp(xx, edges[0], edges[1], 0.0, 1.0)
    return power2_inverse_powern_decreasing(xx_scaled_and_clamped, prefactor=prefactor, powern=powern)