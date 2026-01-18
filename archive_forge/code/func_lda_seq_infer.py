import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def lda_seq_infer(self, corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize):
    """Inference (or E-step) for the lower bound EM optimization.

        This is used to set up the gensim :class:`~gensim.models.ldamodel.LdaModel` to be used for each time-slice.
        It also allows for Document Influence Model code to be written in.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        topic_suffstats : numpy.ndarray
            Sufficient statistics for time slice 0, used for initializing the model if `initialize == 'own'`,
            expected shape (`self.vocab_len`, `num_topics`).
        gammas : numpy.ndarray
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lhoods : list of float
            The total log probability lower bound for each topic. Corresponds to the phi variational parameters in the
            linked paper.
        iter_ : int
            Current iteration.
        lda_inference_max_iter : int
            Maximum number of iterations for the inference step of LDA.
        chunksize : int
            Number of documents to be processed in each chunk.

        Returns
        -------
        (float, list of float)
            The first value is the highest lower bound for the true posterior.
            The second value is the list of optimized dirichlet variational parameters for the approximation of
            the posterior.

        """
    num_topics = self.num_topics
    vocab_len = self.vocab_len
    bound = 0.0
    lda = ldamodel.LdaModel(num_topics=num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)
    lda.topics = np.zeros((vocab_len, num_topics))
    ldapost = LdaPost(max_doc_len=self.max_doc_len, num_topics=num_topics, lda=lda)
    model = 'DTM'
    if model == 'DTM':
        bound, gammas = self.inferDTMseq(corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize)
    elif model == 'DIM':
        self.InfluenceTotalFixed(corpus)
        bound, gammas = self.inferDIMseq(corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize)
    return (bound, gammas)