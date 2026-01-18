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
def evaluate_test_corpus(self, corpus):
    """Evaluates the model on test corpus.

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            Test corpus in BoW format.

        Returns
        -------
        float
            The value of total likelihood obtained by evaluating the model for all documents in the test corpus.

        """
    logger.info('TEST: evaluating test corpus')
    if self.lda_alpha is None or self.lda_beta is None:
        self.lda_alpha, self.lda_beta = self.hdp_to_lda()
    score = 0.0
    total_words = 0
    for i, doc in enumerate(corpus):
        if len(doc) > 0:
            doc_word_ids, doc_word_counts = zip(*doc)
            likelihood, gamma = lda_e_step(doc_word_ids, doc_word_counts, self.lda_alpha, self.lda_beta)
            theta = gamma / np.sum(gamma)
            lda_betad = self.lda_beta[:, doc_word_ids]
            log_predicts = np.log(np.dot(theta, lda_betad))
            doc_score = sum(log_predicts) / len(doc)
            logger.info('TEST: %6d    %.5f', i, doc_score)
            score += likelihood
            total_words += sum(doc_word_counts)
    logger.info('TEST: average score: %.5f, total score: %.5f,  test docs: %d', score / total_words, score, len(corpus))
    return score