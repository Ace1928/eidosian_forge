import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def make_lda_seq_slice(self, lda, time):
    """Update the LDA model topic-word values using time slices.

        Parameters
        ----------

        lda : :class:`~gensim.models.ldamodel.LdaModel`
            The stationary model to be updated
        time : int
            The time slice assigned to the stationary model.

        Returns
        -------
        lda : :class:`~gensim.models.ldamodel.LdaModel`
            The stationary model updated to reflect the passed time slice.

        """
    for k in range(self.num_topics):
        lda.topics[:, k] = self.topic_chains[k].e_log_prob[:, time]
    lda.alpha = np.copy(self.alphas)
    return lda