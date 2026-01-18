from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def fitonce(self, start=None, weights=None, has_optimal_weights=False):
    """fit without estimating an optimal weighting matrix and return results

        This is a convenience function that calls fitgmm and covparams with
        a given weight matrix or the identity weight matrix.
        This is useful if the optimal weight matrix is know (or is analytically
        given) or if an optimal weight matrix cannot be calculated.

        (Developer Notes: this function could go into GMM, but is needed in this
        class, at least at the moment.)

        Parameters
        ----------


        Returns
        -------
        results : GMMResult instance
            result instance with params and _cov_params attached

        See Also
        --------
        fitgmm
        cov_params

        """
    if weights is None:
        weights = np.eye(self.nmoms)
    params = self.fitgmm(start=start)
    self.results.params = params
    self.results.wargs = {}
    self.results.options_other = {'weights_method': 'cov'}
    _cov_params = self.results.cov_params(weights=weights, has_optimal_weights=has_optimal_weights)
    self.results.weights = weights
    self.results.jval = self.gmmobjective(params, weights)
    self.results.options_other.update({'has_optimal_weights': has_optimal_weights})
    return self.results