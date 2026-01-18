import numpy as np
from scipy.special import erf
def aitchison_aitken_cdf(h, Xi, x_u):
    x_u = int(x_u)
    Xi_vals = np.unique(Xi)
    ordered = np.zeros(Xi.size)
    num_levels = Xi_vals.size
    for x in Xi_vals:
        if x <= x_u:
            ordered += aitchison_aitken(h, Xi, x, num_levels=num_levels)
    return ordered