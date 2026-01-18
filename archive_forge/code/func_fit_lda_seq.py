import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def fit_lda_seq(self, corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize):
    """Fit a LDA Sequence model (DTM).

        This method will iteratively setup LDA models and perform EM steps until the sufficient statistics convergence,
        or until the maximum number of iterations is reached. Because the true posterior is intractable, an
        appropriately tight lower bound must be used instead. This function will optimize this bound, by minimizing
        its true Kullback-Liebler Divergence with the true posterior.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        lda_inference_max_iter : int
            Maximum number of iterations for the inference step of LDA.
        em_min_iter : int
            Minimum number of time slices to be inspected.
        em_max_iter : int
            Maximum number of time slices to be inspected.
        chunksize : int
            Number of documents to be processed in each chunk.

        Returns
        -------
        float
            The highest lower bound for the true posterior produced after all iterations.

       """
    LDASQE_EM_THRESHOLD = 0.0001
    LOWER_ITER = 10
    ITER_MULT_LOW = 2
    MAX_ITER = 500
    num_topics = self.num_topics
    vocab_len = self.vocab_len
    data_len = self.num_time_slices
    corpus_len = self.corpus_len
    bound = 0
    convergence = LDASQE_EM_THRESHOLD + 1
    iter_ = 0
    while iter_ < em_min_iter or (convergence > LDASQE_EM_THRESHOLD and iter_ <= em_max_iter):
        logger.info(' EM iter %i', iter_)
        logger.info('E Step')
        old_bound = bound
        topic_suffstats = []
        for topic in range(num_topics):
            topic_suffstats.append(np.zeros((vocab_len, data_len)))
        gammas = np.zeros((corpus_len, num_topics))
        lhoods = np.zeros((corpus_len, num_topics + 1))
        bound, gammas = self.lda_seq_infer(corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize)
        self.gammas = gammas
        logger.info('M Step')
        topic_bound = self.fit_lda_seq_topics(topic_suffstats)
        bound += topic_bound
        if bound - old_bound < 0:
            if lda_inference_max_iter < LOWER_ITER:
                lda_inference_max_iter *= ITER_MULT_LOW
            logger.info('Bound went down, increasing iterations to %i', lda_inference_max_iter)
        convergence = np.fabs((bound - old_bound) / old_bound)
        if convergence < LDASQE_EM_THRESHOLD:
            lda_inference_max_iter = MAX_ITER
            logger.info('Starting final iterations, max iter is %i', lda_inference_max_iter)
            convergence = 1.0
        logger.info('iteration %i iteration lda seq bound is %f convergence is %f', iter_, bound, convergence)
        iter_ += 1
    return bound