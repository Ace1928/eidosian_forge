import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.sandbox.infotheo as infotheo
def mutualinfo_kde_2sample(y, x, normed=True):
    """mutual information of two random variables estimated with kde

    """
    nobs = len(x)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    kde_x = gaussian_kde(x.T)(x.T)
    kde_y = gaussian_kde(y.T)(x.T)
    mi_obs = np.log(kde_x) - np.log(kde_y)
    if len(mi_obs) != nobs:
        raise ValueError('Wrong number of observations')
    mi = mi_obs.mean()
    if normed:
        mi_normed = np.sqrt(1.0 - np.exp(-2 * mi))
        return mi_normed
    else:
        return mi