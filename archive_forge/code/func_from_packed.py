import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def from_packed(params, k_fe, k_re, use_sqrt, has_fe):
    """
        Create a MixedLMParams object from packed parameter vector.

        Parameters
        ----------
        params : array_like
            The mode parameters packed into a single vector.
        k_fe : int
            The number of covariates with fixed effects
        k_re : int
            The number of covariates with random effects (excluding
            variance components).
        use_sqrt : bool
            If True, the random effects covariance matrix is provided
            as its Cholesky factor, otherwise the lower triangle of
            the covariance matrix is stored.
        has_fe : bool
            If True, `params` contains fixed effects parameters.
            Otherwise, the fixed effects parameters are set to zero.

        Returns
        -------
        A MixedLMParams object.
        """
    k_re2 = int(k_re * (k_re + 1) / 2)
    if has_fe:
        k_vc = len(params) - k_fe - k_re2
    else:
        k_vc = len(params) - k_re2
    pa = MixedLMParams(k_fe, k_re, k_vc)
    cov_re = np.zeros((k_re, k_re))
    ix = pa._ix
    if has_fe:
        pa.fe_params = params[0:k_fe]
        cov_re[ix] = params[k_fe:k_fe + k_re2]
    else:
        pa.fe_params = np.zeros(k_fe)
        cov_re[ix] = params[0:k_re2]
    if use_sqrt:
        cov_re = np.dot(cov_re, cov_re.T)
    else:
        cov_re = cov_re + cov_re.T - np.diag(np.diag(cov_re))
    pa.cov_re = cov_re
    if k_vc > 0:
        if use_sqrt:
            pa.vcomp = params[-k_vc:] ** 2
        else:
            pa.vcomp = params[-k_vc:]
    else:
        pa.vcomp = np.array([])
    return pa