from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def fitgmm_cu(self, start, optim_method='bfgs', optim_args=None):
    """estimate parameters using continuously updating GMM

        Parameters
        ----------
        start : array_like
            starting values for minimization

        Returns
        -------
        paramest : ndarray
            estimated parameters

        Notes
        -----
        todo: add fixed parameter option, not here ???

        uses scipy.optimize.fmin

        """
    if optim_args is None:
        optim_args = {}
    if optim_method == 'nm':
        optimizer = optimize.fmin
    elif optim_method == 'bfgs':
        optimizer = optimize.fmin_bfgs
        optim_args['fprime'] = self.score_cu
    elif optim_method == 'ncg':
        optimizer = optimize.fmin_ncg
    else:
        raise ValueError('optimizer method not available')
    return optimizer(self.gmmobjective_cu, start, args=(), **optim_args)