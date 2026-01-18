import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-08, lda_inference_max_iter=25, g=None, g3_matrix=None, g4_matrix=None, g5_matrix=None):
    """Posterior inference for lda.

        Parameters
        ----------
        doc_number : int
            The documents number.
        time : int
            Time slice.
        ldaseq : object
            Unused.
        LDA_INFERENCE_CONVERGED : float
            Epsilon value used to check whether the inference step has sufficiently converged.
        lda_inference_max_iter : int
            Maximum number of iterations in the inference step.
        g : object
            Unused. Will be useful when the DIM model is implemented.
        g3_matrix: object
            Unused. Will be useful when the DIM model is implemented.
        g4_matrix: object
            Unused. Will be useful when the DIM model is implemented.
        g5_matrix: object
            Unused. Will be useful when the DIM model is implemented.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.
        """
    self.init_lda_post()
    total = sum((count for word_id, count in self.doc))
    model = 'DTM'
    if model == 'DIM':
        pass
    lhood = self.compute_lda_lhood()
    lhood_old = 0
    converged = 0
    iter_ = 0
    iter_ += 1
    lhood_old = lhood
    self.gamma = self.update_gamma()
    model = 'DTM'
    if model == 'DTM' or sslm is None:
        self.phi, self.log_phi = self.update_phi(doc_number, time)
    elif model == 'DIM' and sslm is not None:
        self.phi, self.log_phi = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)
    lhood = self.compute_lda_lhood()
    converged = np.fabs((lhood_old - lhood) / (lhood_old * total))
    while converged > LDA_INFERENCE_CONVERGED and iter_ <= lda_inference_max_iter:
        iter_ += 1
        lhood_old = lhood
        self.gamma = self.update_gamma()
        model = 'DTM'
        if model == 'DTM' or sslm is None:
            self.phi, self.log_phi = self.update_phi(doc_number, time)
        elif model == 'DIM' and sslm is not None:
            self.phi, self.log_phi = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)
        lhood = self.compute_lda_lhood()
        converged = np.fabs((lhood_old - lhood) / (lhood_old * total))
    return lhood