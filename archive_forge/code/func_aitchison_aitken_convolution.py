import numpy as np
from scipy.special import erf
def aitchison_aitken_convolution(h, Xi, Xj):
    Xi_vals = np.unique(Xi)
    ordered = np.zeros(Xi.size)
    num_levels = Xi_vals.size
    for x in Xi_vals:
        ordered += aitchison_aitken(h, Xi, x, num_levels=num_levels) * aitchison_aitken(h, Xj, x, num_levels=num_levels)
    return ordered