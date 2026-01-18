import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def density_otherwise(y, mu, p, phi):
    theta = _theta(mu, p)
    logd = logW(y, p, phi) - np.log(y) + 1 / phi * (y * theta - kappa(mu, p))
    return np.exp(logd)