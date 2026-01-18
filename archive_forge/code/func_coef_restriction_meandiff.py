from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
def coef_restriction_meandiff(n_coeffs, n_vars=None, position=0):
    reduced = np.eye(n_coeffs) - 1.0 / n_coeffs
    if n_vars is None:
        return reduced
    else:
        full = np.zeros((n_coeffs, n_vars))
        full[:, position:position + n_coeffs] = reduced
        return full