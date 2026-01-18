from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
def coef_restriction_diffbase(n_coeffs, n_vars=None, position=0, base_idx=0):
    reduced = -np.eye(n_coeffs)
    reduced[:, base_idx] = 1
    keep = lrange(n_coeffs)
    del keep[base_idx]
    reduced = np.take(reduced, keep, axis=0)
    if n_vars is None:
        return reduced
    else:
        full = np.zeros((n_coeffs - 1, n_vars))
        full[:, position:position + n_coeffs] = reduced
        return full