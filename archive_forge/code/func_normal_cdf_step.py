from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def normal_cdf_step(xx, mean, scale):
    """

    Args:
        xx:
        mean:
        scale:
    """
    return 0.5 * (1.0 + erf((xx - mean) / (np.sqrt(2.0) * scale)))