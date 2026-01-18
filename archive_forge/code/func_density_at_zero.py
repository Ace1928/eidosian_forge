import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def density_at_zero(y, mu, p, phi):
    return np.exp(-mu ** (2 - p) / (phi * (2 - p)))