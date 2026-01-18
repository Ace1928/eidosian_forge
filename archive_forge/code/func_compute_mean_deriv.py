import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_mean_deriv(self, word, time, deriv):
    """Helper functions for optimizing a function.

        Compute the derivative of:

        .. :math::

            E[\x08eta_{t,w}]/d obs_{s,w} for t = 1:T.

        Parameters
        ----------
        word : int
            The word's ID.
        time : int
            The time slice.
        deriv : list of float
            Derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """
    T = self.num_time_slices
    fwd_variance = self.variance[word]
    deriv[0] = 0
    for t in range(1, T + 1):
        if self.obs_variance > 0.0:
            w = self.obs_variance / (fwd_variance[t - 1] + self.chain_variance + self.obs_variance)
        else:
            w = 0.0
        val = w * deriv[t - 1]
        if time == t - 1:
            val += 1 - w
        deriv[t] = val
    for t in range(T - 1, -1, -1):
        if self.chain_variance == 0.0:
            w = 0.0
        else:
            w = self.chain_variance / (fwd_variance[t] + self.chain_variance)
        deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]
    return deriv