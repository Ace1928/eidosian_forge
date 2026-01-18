import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):
    """Derivation of obs which is used in derivative function `df_obs` while optimizing.

        Parameters
        ----------
        word : int
            The word's ID.
        word_counts : list of int
            Total word counts for each time slice.
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.
        mean_deriv_mtx : list of float
            Mean derivative for each time slice.
        deriv : list of float
            Mean derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """
    init_mult = 1000
    T = self.num_time_slices
    mean = self.mean[word]
    variance = self.variance[word]
    self.temp_vect = np.zeros(T)
    for u in range(T):
        self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)
    for t in range(T):
        mean_deriv = mean_deriv_mtx[t]
        term1 = 0
        term2 = 0
        term3 = 0
        term4 = 0
        for u in range(1, T + 1):
            mean_u = mean[u]
            mean_u_prev = mean[u - 1]
            dmean_u = mean_deriv[u]
            dmean_u_prev = mean_deriv[u - 1]
            term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
            term2 += (word_counts[u - 1] - totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1]) * dmean_u
            model = 'DTM'
            if model == 'DIM':
                pass
        if self.chain_variance:
            term1 = -(term1 / self.chain_variance)
            term1 = term1 - mean[0] * mean_deriv[0] / (init_mult * self.chain_variance)
        else:
            term1 = 0.0
        deriv[t] = term1 + term2 + term3 + term4
    return deriv