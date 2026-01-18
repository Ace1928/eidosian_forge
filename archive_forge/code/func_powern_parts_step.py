from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def powern_parts_step(xx, edges=None, inverse=False, nn=2):
    """

    Args:
        xx:
        edges:
        inverse:
        nn:
    """
    if edges is None:
        aa = np.power(0.5, 1.0 - nn)
        xx_clipped = np.clip(xx, 0.0, 1.0)
        if np.mod(nn, 2) == 0:
            if inverse:
                return 1.0 - np.where(xx_clipped < 0.5, aa * np.power(xx_clipped, nn), 1.0 - aa * np.power(xx_clipped - 1.0, nn))
            return np.where(xx_clipped < 0.5, aa * np.power(xx_clipped, nn), 1.0 - aa * np.power(xx_clipped - 1.0, nn))
        if inverse:
            return 1.0 - np.where(xx_clipped < 0.5, aa * np.power(xx_clipped, nn), 1.0 + aa * np.power(xx_clipped - 1.0, nn))
        return np.where(xx_clipped < 0.5, aa * np.power(xx_clipped, nn), 1.0 + aa * np.power(xx_clipped - 1.0, nn))
    xx_scaled_and_clamped = scale_and_clamp(xx, edges[0], edges[1], 0.0, 1.0)
    return powern_parts_step(xx_scaled_and_clamped, inverse=inverse, nn=nn)