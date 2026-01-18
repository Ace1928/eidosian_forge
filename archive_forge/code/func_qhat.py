from statsmodels.compat.python import lrange, lmap
import math
import scipy.stats
from scipy.optimize import leastsq
import numpy as np
from numpy.random import random
def qhat(a, p, r, v):
    p_ = (1.0 + p) / 2.0
    f = a[0] * np.log(r - 1.0) + a[1] * np.log(r - 1.0) ** 2 + a[2] * np.log(r - 1.0) ** 3 + a[3] * np.log(r - 1.0) ** 4
    for i, r_ in enumerate(r):
        if r_ == 3:
            f[i] += -0.002 / (1.0 + 12.0 * _phi(p) ** 2)
            if v <= 4.364:
                f[i] += 1.0 / 517.0 - 1.0 / (312.0 * v)
            else:
                f[i] += 1.0 / (191.0 * v)
    return math.sqrt(2) * (f - 1.0) * _tinv(p_, v)