import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def inferDTMseq(self, corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize):
    """Compute the likelihood of a sequential corpus under an LDA seq model, and reports the likelihood bound.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        topic_suffstats : numpy.ndarray
            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).
        gammas : numpy.ndarray
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lhoods : list of float of length `self.num_topics`
            The total log probability bound for each topic. Corresponds to phi from the linked paper.
        lda : :class:`~gensim.models.ldamodel.LdaModel`
            The trained LDA model of the previous iteration.
        ldapost : :class:`~gensim.models.ldaseqmodel.LdaPost`
            Posterior probability variables for the given LDA model. This will be used as the true (but intractable)
            posterior.
        iter_ : int
            The current iteration.
        bound : float
            The LDA bound produced after all iterations.
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
    doc_index = 0
    time = 0
    doc_num = 0
    lda = self.make_lda_seq_slice(lda, time)
    time_slice = np.cumsum(np.array(self.time_slice))
    for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize)):
        for doc in chunk:
            if doc_index > time_slice[time]:
                time += 1
                lda = self.make_lda_seq_slice(lda, time)
                doc_num = 0
            gam = gammas[doc_index]
            lhood = lhoods[doc_index]
            ldapost.gamma = gam
            ldapost.lhood = lhood
            ldapost.doc = doc
            if iter_ == 0:
                doc_lhood = LdaPost.fit_lda_post(ldapost, doc_num, time, None, lda_inference_max_iter=lda_inference_max_iter)
            else:
                doc_lhood = LdaPost.fit_lda_post(ldapost, doc_num, time, self, lda_inference_max_iter=lda_inference_max_iter)
            if topic_suffstats is not None:
                topic_suffstats = LdaPost.update_lda_seq_ss(ldapost, time, doc, topic_suffstats)
            gammas[doc_index] = ldapost.gamma
            bound += doc_lhood
            doc_index += 1
            doc_num += 1
    return (bound, gammas)