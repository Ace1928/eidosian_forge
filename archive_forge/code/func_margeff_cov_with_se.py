from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def margeff_cov_with_se(model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J):
    """
    See margeff_cov_params.

    Same function but returns both the covariance of the marginal effects
    and their standard errors.
    """
    cov_me = margeff_cov_params(model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J)
    return (cov_me, np.sqrt(np.diag(cov_me)))