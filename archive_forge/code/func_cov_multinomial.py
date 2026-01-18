import numpy as np
from scipy import stats
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.effect_size import _noncentrality_chisquare
def cov_multinomial(probs):
    """covariance matrix of multinomial distribution

    This is vectorized with choices along last axis.

    cov = diag(probs) - outer(probs, probs)

    """
    k = probs.shape[-1]
    di = np.diag_indices(k, 2)
    cov = probs[..., None] * probs[..., None, :]
    cov *= -1
    cov[..., di[0], di[1]] += probs
    return cov