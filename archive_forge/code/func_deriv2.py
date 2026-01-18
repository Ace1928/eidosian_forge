import warnings
import numpy as np
from scipy.special import expm1, gamma
def deriv2(self, t, *args):
    t = np.asarray(t)
    return 1.0 / t ** 2