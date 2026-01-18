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
def suggested_lda_model(self):
    """Get a trained ldamodel object which is closest to the current hdp model.

        The `num_topics=m_T`, so as to preserve the matrices shapes when we assign alpha and beta.

        Returns
        -------
        :class:`~gensim.models.ldamodel.LdaModel`
            Closest corresponding LdaModel to current HdpModel.

        """
    alpha, beta = self.hdp_to_lda()
    ldam = ldamodel.LdaModel(num_topics=self.m_T, alpha=alpha, id2word=self.id2word, random_state=self.random_state, dtype=np.float64)
    ldam.expElogbeta[:] = beta
    return ldam