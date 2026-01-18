import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_lda_lhood(self):
    """Compute the log likelihood bound.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.

        """
    num_topics = self.lda.num_topics
    gamma_sum = np.sum(self.gamma)
    lhood = gammaln(np.sum(self.lda.alpha)) - gammaln(gamma_sum)
    self.lhood[num_topics] = lhood
    digsum = digamma(gamma_sum)
    model = 'DTM'
    for k in range(num_topics):
        e_log_theta_k = digamma(self.gamma[k]) - digsum
        lhood_term = (self.lda.alpha[k] - self.gamma[k]) * e_log_theta_k + gammaln(self.gamma[k]) - gammaln(self.lda.alpha[k])
        n = 0
        for word_id, count in self.doc:
            if self.phi[n][k] > 0:
                lhood_term += count * self.phi[n][k] * (e_log_theta_k + self.lda.topics[word_id][k] - self.log_phi[n][k])
            n += 1
        self.lhood[k] = lhood_term
        lhood += lhood_term
    return lhood