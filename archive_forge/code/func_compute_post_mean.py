import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_post_mean(self, word, chain_variance):
    """Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        Notes
        -----
        This function essentially computes E[\x08eta_{t,w}] for t = 1:T.

        .. :math::

            Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
            = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +
            (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

        .. :math::

            Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
            = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +
            (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the mean of each word in each time slice, the second value is the
            inferred posterior mean for the same pairs.

        """
    T = self.num_time_slices
    obs = self.obs[word]
    fwd_variance = self.fwd_variance[word]
    mean = self.mean[word]
    fwd_mean = self.fwd_mean[word]
    fwd_mean[0] = 0
    for t in range(1, T + 1):
        c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
        fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]
    mean[T] = fwd_mean[T]
    for t in range(T - 1, -1, -1):
        if chain_variance == 0.0:
            c = 0.0
        else:
            c = chain_variance / (fwd_variance[t] + chain_variance)
        mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
    return (mean, fwd_mean)