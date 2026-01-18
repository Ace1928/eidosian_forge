import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def series_density(y, mu, p, phi):
    density = _lazywhere(np.array(y) > 0, (y, mu, p, phi), f=density_otherwise, f2=density_at_zero)
    return density