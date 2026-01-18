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
def doc_e_step(self, ss, Elogsticks_1st, unique_words, doc_word_ids, doc_word_counts, var_converge):
    """Performs E step for a single doc.

        Parameters
        ----------
        ss : :class:`~gensim.models.hdpmodel.SuffStats`
            Stats for all document(s) in the chunk.
        Elogsticks_1st : numpy.ndarray
            Computed Elogsticks value by stick-breaking process.
        unique_words : dict of (int, int)
            Number of unique words in the chunk.
        doc_word_ids : iterable of int
            Word ids of for a single document.
        doc_word_counts : iterable of int
            Word counts of all words in a single document.
        var_converge : float
            Lower bound on the right side of convergence. Used when updating variational parameters for a single
            document.

        Returns
        -------
        float
            Computed value of likelihood for a single document.

        """
    chunkids = [unique_words[id] for id in doc_word_ids]
    Elogbeta_doc = self.m_Elogbeta[:, doc_word_ids]
    v = np.zeros((2, self.m_K - 1))
    v[0] = 1.0
    v[1] = self.m_alpha
    phi = np.ones((len(doc_word_ids), self.m_K)) * 1.0 / self.m_K
    likelihood = 0.0
    old_likelihood = -1e+200
    converge = 1.0
    iter = 0
    max_iter = 100
    while iter < max_iter and (converge < 0.0 or converge > var_converge):
        if iter < 3:
            var_phi = np.dot(phi.T, (Elogbeta_doc * doc_word_counts).T)
            log_var_phi, log_norm = matutils.ret_log_normalize_vec(var_phi)
            var_phi = np.exp(log_var_phi)
        else:
            var_phi = np.dot(phi.T, (Elogbeta_doc * doc_word_counts).T) + Elogsticks_1st
            log_var_phi, log_norm = matutils.ret_log_normalize_vec(var_phi)
            var_phi = np.exp(log_var_phi)
        if iter < 3:
            phi = np.dot(var_phi, Elogbeta_doc).T
            log_phi, log_norm = matutils.ret_log_normalize_vec(phi)
            phi = np.exp(log_phi)
        else:
            phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
            log_phi, log_norm = matutils.ret_log_normalize_vec(phi)
            phi = np.exp(log_phi)
        phi_all = phi * np.array(doc_word_counts)[:, np.newaxis]
        v[0] = 1.0 + np.sum(phi_all[:, :self.m_K - 1], 0)
        phi_cum = np.flipud(np.sum(phi_all[:, 1:], 0))
        v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
        Elogsticks_2nd = expect_log_sticks(v)
        likelihood = 0.0
        likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)
        log_alpha = np.log(self.m_alpha)
        likelihood += (self.m_K - 1) * log_alpha
        dig_sum = psi(np.sum(v, 0))
        likelihood += np.sum((np.array([1.0, self.m_alpha])[:, np.newaxis] - v) * (psi(v) - dig_sum))
        likelihood -= np.sum(gammaln(np.sum(v, 0))) - np.sum(gammaln(v))
        likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)
        likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc_word_counts))
        converge = (likelihood - old_likelihood) / abs(old_likelihood)
        old_likelihood = likelihood
        if converge < -1e-06:
            logger.warning('likelihood is decreasing!')
        iter += 1
    ss.m_var_sticks_ss += np.sum(var_phi, 0)
    ss.m_var_beta_ss[:, chunkids] += np.dot(var_phi.T, phi.T * doc_word_counts)
    return likelihood