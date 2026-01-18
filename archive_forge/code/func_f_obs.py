import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def f_obs(x, *args):
    """Function which we are optimising for minimizing obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The value of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args
    init_mult = 1000
    T = len(x)
    val = 0
    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0
    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)
    mean = sslm.mean[word]
    variance = sslm.variance[word]
    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]
        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * np.exp(mean_t + variance[t] / 2) / sslm.zeta[t - 1]
        model = 'DTM'
        if model == 'DIM':
            pass
    if sslm.chain_variance > 0.0:
        term1 = -(term1 / (2 * sslm.chain_variance))
        term1 = term1 - mean[0] * mean[0] / (2 * init_mult * sslm.chain_variance)
    else:
        term1 = 0.0
    final = -(term1 + term2 + term3 + term4)
    return final