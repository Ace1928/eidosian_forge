from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def fitgmm(self, start, weights=None, optim_method=None, **kwds):
    """estimate parameters using GMM for linear model

        Uses closed form expression instead of nonlinear optimizers

        Parameters
        ----------
        start : not used
            starting values for minimization, not used, only for consistency
            of method signature
        weights : ndarray
            weighting matrix for moment conditions. If weights is None, then
            the identity matrix is used
        optim_method : not used,
            optimization method, not used, only for consistency of method
            signature
        **kwds : keyword arguments
            not used, will be silently ignored (for compatibility with generic)


        Returns
        -------
        paramest : ndarray
            estimated parameters

        """
    if weights is None:
        weights = self.start_weights(inv=False)
    y, x, z = (self.endog, self.exog, self.instrument)
    zTx = np.dot(z.T, x)
    zTy = np.dot(z.T, y)
    part0 = zTx.T.dot(weights)
    part1 = part0.dot(zTx)
    part2 = part0.dot(zTy)
    params = np.linalg.pinv(part1).dot(part2)
    return params