import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def score_factor(self, params, endog=None):
    """Derivative of loglikelihood function w.r.t. linear predictors.

        This needs to be multiplied with the exog to obtain the score_obs.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.

        Returns
        -------
        score_factor : ndarray, 2-D
            A 2d weight vector used in the calculation of the score_obs.

        Notes
        -----
        The score_obs can be obtained from score_factor ``sf`` using

            - d1 = sf[:, :1] * exog
            - d2 = sf[:, 1:2] * exog_precision

        """
    from scipy import special
    digamma = special.psi
    y = self.endog if endog is None else endog
    X, Z = (self.exog, self.exog_precision)
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
    sf1 = phi * t * (ystar - mustar)
    sf2 = h * (mu * (ystar - mustar) + yt - mut)
    return (sf1, sf2)