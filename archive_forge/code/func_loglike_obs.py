import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
    """
        The log-likelihood function for each observation in terms of the fitted
        mean response for the Tweedie distribution.

        Parameters
        ----------
        endog : ndarray
            Usually the endogenous response variable.
        mu : ndarray
            Usually but not always the fitted mean response variable.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float
            The scale parameter. The default is 1.

        Returns
        -------
        ll_i : float
            The value of the loglikelihood evaluated at
            (endog, mu, var_weights, scale) as defined below.

        Notes
        -----
        If eql is True, the Extended Quasi-Likelihood is used.  At present,
        this method returns NaN if eql is False.  When the actual likelihood
        is implemented, it will be accessible by setting eql to False.

        References
        ----------
        R Kaas (2005).  Compound Poisson Distributions and GLM's -- Tweedie's
        Distribution.
        https://core.ac.uk/download/pdf/6347266.pdf#page=11

        JA Nelder, D Pregibon (1987).  An extended quasi-likelihood function.
        Biometrika 74:2, pp 221-232.  https://www.jstor.org/stable/2336136
        """
    p = self.var_power
    endog = np.atleast_1d(endog)
    if p == 1:
        return Poisson().loglike_obs(endog=endog, mu=mu, var_weights=var_weights, scale=scale)
    elif p == 2:
        return Gamma().loglike_obs(endog=endog, mu=mu, var_weights=var_weights, scale=scale)
    if not self.eql:
        if p < 1 or p > 2:
            return np.nan
        if SP_LT_17:
            return np.nan
        scale = scale / var_weights
        theta = mu ** (1 - p) / (1 - p)
        kappa = mu ** (2 - p) / (2 - p)
        alpha = (2 - p) / (1 - p)
        ll_obs = (endog * theta - kappa) / scale
        idx = endog > 0
        if np.any(idx):
            if not np.isscalar(endog):
                endog = endog[idx]
            if not np.isscalar(scale):
                scale = scale[idx]
            x = ((p - 1) * scale / endog) ** alpha
            x /= (2 - p) * scale
            wb = special.wright_bessel(-alpha, 0, x)
            ll_obs[idx] += np.log(1 / endog * wb)
        return ll_obs
    else:
        llf = np.log(2 * np.pi * scale) + p * np.log(endog)
        llf -= np.log(var_weights)
        llf /= -2
        u = endog ** (2 - p) - (2 - p) * endog * mu ** (1 - p) + (1 - p) * mu ** (2 - p)
        u *= var_weights / (scale * (1 - p) * (2 - p))
    return llf - u