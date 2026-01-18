from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
def coef_restriction_diffseq(n_coeffs, degree=1, n_vars=None, position=0, base_idx=0):
    if degree == 1:
        diff_coeffs = [-1, 1]
        n_points = 2
    elif degree > 1:
        from scipy import misc
        n_points = next_odd(degree + 1)
        diff_coeffs = misc.central_diff_weights(n_points, ndiv=degree)
    dff = np.concatenate((diff_coeffs, np.zeros(n_coeffs - len(diff_coeffs))))
    from scipy import linalg
    reduced = linalg.toeplitz(dff, np.zeros(n_coeffs - len(diff_coeffs) + 1)).T
    if n_vars is None:
        return reduced
    else:
        full = np.zeros((n_coeffs - 1, n_vars))
        full[:, position:position + n_coeffs] = reduced
        return full