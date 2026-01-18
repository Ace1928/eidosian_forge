import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
def corr_ar(k_vars, ar):
    """create autoregressive correlation matrix

    This might be MA, not AR, process if used for residual process - check

    Parameters
    ----------
    ar : array_like, 1d
        AR lag-polynomial including 1 for lag 0


    """
    from scipy.linalg import toeplitz
    if len(ar) < k_vars:
        ar_ = np.zeros(k_vars)
        ar_[:len(ar)] = ar
        ar = ar_
    return toeplitz(ar)