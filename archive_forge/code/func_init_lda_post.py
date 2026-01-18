import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def init_lda_post(self):
    """Initialize variational posterior. """
    total = sum((count for word_id, count in self.doc))
    self.gamma.fill(self.lda.alpha[0] + float(total) / self.lda.num_topics)
    self.phi[:len(self.doc), :] = 1.0 / self.lda.num_topics