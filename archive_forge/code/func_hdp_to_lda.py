from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
import logging
import time
import warnings
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models import basemodel, ldamodel
from gensim.utils import deprecated
def hdp_to_lda(self):
    """Get corresponding alpha and beta values of a LDA almost equivalent to current HDP.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Alpha and Beta arrays.

        """
    sticks = self.m_var_sticks[0] / (self.m_var_sticks[0] + self.m_var_sticks[1])
    alpha = np.zeros(self.m_T)
    left = 1.0
    for i in range(0, self.m_T - 1):
        alpha[i] = sticks[i] * left
        left = left - alpha[i]
    alpha[self.m_T - 1] = left
    alpha *= self.m_alpha
    beta = (self.m_lambda + self.m_eta) / (self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])
    return (alpha, beta)