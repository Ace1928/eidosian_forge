from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def cosinus_step(xx, edges=None, inverse=False):
    """

    Args:
        xx:
        edges:
        inverse:
    """
    if edges is None:
        xx_clipped = np.clip(xx, 0.0, 1.0)
        if inverse:
            return (np.cos(xx_clipped * np.pi) + 1.0) / 2.0
        return 1.0 - (np.cos(xx_clipped * np.pi) + 1.0) / 2.0
    xx_scaled_and_clamped = scale_and_clamp(xx, edges[0], edges[1], 0.0, 1.0)
    return cosinus_step(xx_scaled_and_clamped, inverse=inverse)