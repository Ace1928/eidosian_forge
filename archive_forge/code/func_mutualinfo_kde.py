import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.sandbox.infotheo as infotheo
def mutualinfo_kde(y, x, normed=True):
    """mutual information of two random variables estimated with kde

    """
    nobs = len(x)
    if not len(y) == nobs:
        raise ValueError('both data arrays need to have the same size')
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yx = np.vstack((y, x))
    kde_x = gaussian_kde(x)(x)
    kde_y = gaussian_kde(y)(y)
    kde_yx = gaussian_kde(yx)(yx)
    mi_obs = np.log(kde_yx) - np.log(kde_x) - np.log(kde_y)
    mi = mi_obs.sum() / nobs
    if normed:
        mi_normed = np.sqrt(1.0 - np.exp(-2 * mi))
        return mi_normed
    else:
        return mi