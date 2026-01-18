from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def renyientropy(px, alpha=1, logbase=2, measure='R'):
    """
    Renyi's generalized entropy

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X.  Note that
        px is assumed to be a proper probability distribution.
    logbase : int or np.e, optional
        Default is 2 (bits)
    alpha : float or inf
        The order of the entropy.  The default is 1, which in the limit
        is just Shannon's entropy.  2 is Renyi (Collision) entropy.  If
        the string "inf" or numpy.inf is specified the min-entropy is returned.
    measure : str, optional
        The type of entropy measure desired.  'R' returns Renyi entropy
        measure.  'T' returns the Tsallis entropy measure.

    Returns
    -------
    1/(1-alpha)*log(sum(px**alpha))

    In the limit as alpha -> 1, Shannon's entropy is returned.

    In the limit as alpha -> inf, min-entropy is returned.
    """
    if not _isproperdist(px):
        raise ValueError('px is not a proper probability distribution')
    alpha = float(alpha)
    if alpha == 1:
        genent = shannonentropy(px)
        if logbase != 2:
            return logbasechange(2, logbase) * genent
        return genent
    elif 'inf' in str(alpha).lower() or alpha == np.inf:
        return -np.log(np.max(px))
    px = px ** alpha
    genent = np.log(px.sum())
    if logbase == 2:
        return 1 / (1 - alpha) * genent
    else:
        return 1 / (1 - alpha) * logbasechange(2, logbase) * genent