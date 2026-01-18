import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def df_obs(x, *args):
    """Derivative of the objective function which optimises obs.

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
        The derivative of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args
    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)
    model = 'DTM'
    if model == 'DTM':
        deriv = sslm.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)
    elif model == 'DIM':
        deriv = sslm.compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)
    return np.negative(deriv)