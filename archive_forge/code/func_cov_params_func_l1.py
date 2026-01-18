from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
def cov_params_func_l1(self, likelihood_model, xopt, retvals):
    """
        Computes cov_params on a reduced parameter space
        corresponding to the nonzero parameters resulting from the
        l1 regularized fit.

        Returns a full cov_params matrix, with entries corresponding
        to zero'd values set to np.nan.
        """
    H = likelihood_model.hessian(xopt)
    trimmed = retvals['trimmed']
    nz_idx = np.nonzero(~trimmed)[0]
    nnz_params = (~trimmed).sum()
    if nnz_params > 0:
        H_restricted = H[nz_idx[:, None], nz_idx]
        H_restricted_inv = np.linalg.inv(-H_restricted)
    else:
        H_restricted_inv = np.zeros(0)
    cov_params = np.nan * np.ones(H.shape)
    cov_params[nz_idx[:, None], nz_idx] = H_restricted_inv
    return cov_params