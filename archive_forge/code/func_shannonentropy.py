from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def shannonentropy(px, logbase=2):
    """
    This is Shannon's entropy

    Parameters
    ----------
    logbase, int or np.e
        The base of the log
    px : 1d or 2d array_like
        Can be a discrete probability distribution, a 2d joint distribution,
        or a sequence of probabilities.

    Returns
    -----
    For log base 2 (bits) given a discrete distribution
        H(p) = sum(px * log2(1/px) = -sum(pk*log2(px)) = E[log2(1/p(X))]

    For log base 2 (bits) given a joint distribution
        H(px,py) = -sum_{k,j}*w_{kj}log2(w_{kj})

    Notes
    -----
    shannonentropy(0) is defined as 0
    """
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError('px does not define proper distribution')
    entropy = -np.sum(np.nan_to_num(px * np.log2(px)))
    if logbase != 2:
        return logbasechange(2, logbase) * entropy
    else:
        return entropy