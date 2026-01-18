import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
def ar2transform(x, arcoefs):
    """

    (Greene eq 12-30)
    """
    a1, a2 = arcoefs
    y = np.zeros_like(x)
    y[0] = np.sqrt((1 + a2) * ((1 - a2) ** 2 - a1 ** 2) / (1 - a2)) * x[0]
    y[1] = np.sqrt(1 - a2 ** 2) * x[2] - a1 * np.sqrt(1 - a1 ** 2) / (1 - a2) * x[1]
    y[2:] = x[2:] - a1 * x[1:-1] - a2 * x[:-2]
    return y