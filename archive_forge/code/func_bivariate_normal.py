import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
def bivariate_normal(x, mu, cov):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.

    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    X, Y = np.transpose(x)
    mux, muy = mu
    sigmax, sigmaxy, tmp, sigmay = np.ravel(cov)
    sigmax, sigmay = (np.sqrt(sigmax), np.sqrt(sigmay))
    Xmu = X - mux
    Ymu = Y - muy
    rho = sigmaxy / (sigmax * sigmay)
    z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
    return np.exp(-z / (2 * (1 - rho ** 2))) / denom