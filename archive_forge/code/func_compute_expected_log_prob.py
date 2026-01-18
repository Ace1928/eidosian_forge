import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_expected_log_prob(self):
    """Compute the expected log probability given values of m.

        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
        The below implementation is the result of solving the equation and is implemented as in the original
        Blei DTM code.

        Returns
        -------
        numpy.ndarray of float
            The expected value for the log probabilities for each word and time slice.

        """
    for (w, t), val in np.ndenumerate(self.e_log_prob):
        self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
    return self.e_log_prob