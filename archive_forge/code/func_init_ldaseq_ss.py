import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def init_ldaseq_ss(self, topic_chain_variance, topic_obs_variance, alpha, init_suffstats):
    """Initialize State Space Language Model, topic-wise.

        Parameters
        ----------
        topic_chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve.
        topic_obs_variance : float
            Observed variance used to approximate the true and forward variance as shown in
            `David M. Blei, John D. Lafferty: "Dynamic Topic Models"
            <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.
        alpha : float
            The prior probability for the model.
        init_suffstats : numpy.ndarray
            Sufficient statistics used for initializing the model, expected shape (`self.vocab_len`, `num_topics`).

        """
    self.alphas = alpha
    for k, chain in enumerate(self.topic_chains):
        sstats = init_suffstats[:, k]
        sslm.sslm_counts_init(chain, topic_obs_variance, topic_chain_variance, sstats)