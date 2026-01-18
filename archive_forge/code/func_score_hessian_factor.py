import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def score_hessian_factor(self, params, return_hessian=False, observed=True):
    """Derivatives of loglikelihood function w.r.t. linear predictors.

        This calculates score and hessian factors at the same time, because
        there is a large overlap in calculations.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        return_hessian : bool
            If False, then only score_factors are returned
            If True, the both score and hessian factors are returned
        observed : bool
            If True, then the observed Hessian is returned (default).
            If False, then the expected information matrix is returned.

        Returns
        -------
        score_factor : ndarray, 2-D
            A 2d weight vector used in the calculation of the score_obs.
        (-jbb, -jbg, -jgg) : tuple
            A tuple with 3 hessian factors, corresponding to the upper
            triangle of the Hessian matrix.
            TODO: check why there are minus
        """
    from scipy import special
    digamma = special.psi
    y, X, Z = (self.endog, self.exog, self.exog_precision)
    nz = Z.shape[1]
    Xparams = params[:-nz]
    Zparams = params[-nz:]
    mu = self.link.inverse(np.dot(X, Xparams))
    phi = self.link_precision.inverse(np.dot(Z, Zparams))
    eps_lb = 1e-200
    alpha = np.clip(mu * phi, eps_lb, np.inf)
    beta = np.clip((1 - mu) * phi, eps_lb, np.inf)
    ystar = np.log(y / (1.0 - y))
    dig_beta = digamma(beta)
    mustar = digamma(alpha) - dig_beta
    yt = np.log(1 - y)
    mut = dig_beta - digamma(phi)
    t = 1.0 / self.link.deriv(mu)
    h = 1.0 / self.link_precision.deriv(phi)
    ymu_star = ystar - mustar
    sf1 = phi * t * ymu_star
    sf2 = h * (mu * ymu_star + yt - mut)
    if return_hessian:
        trigamma = lambda x: special.polygamma(1, x)
        trig_beta = trigamma(beta)
        var_star = trigamma(alpha) + trig_beta
        var_t = trig_beta - trigamma(phi)
        c = -trig_beta
        s = self.link.deriv2(mu)
        q = self.link_precision.deriv2(phi)
        jbb = phi * t * var_star
        if observed:
            jbb += s * t ** 2 * ymu_star
        jbb *= t * phi
        jbg = phi * t * h * (mu * var_star + c)
        if observed:
            jbg -= ymu_star * t * h
        jgg = h ** 2 * (mu ** 2 * var_star + 2 * mu * c + var_t)
        if observed:
            jgg += (mu * ymu_star + yt - mut) * q * h ** 3
        return ((sf1, sf2), (-jbb, -jbg, -jgg))
    else:
        return (sf1, sf2)