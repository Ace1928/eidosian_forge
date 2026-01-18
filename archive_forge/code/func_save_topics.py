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
@deprecated('This method will be removed in 4.0.0, use `save` instead.')
def save_topics(self, doc_count=None):
    """Save discovered topics.

        Warnings
        --------
        This method is deprecated, use :meth:`~gensim.models.hdpmodel.HdpModel.save` instead.

        Parameters
        ----------
        doc_count : int, optional
            Indicates number of documents finished processing and are to be saved.

        """
    if not self.outputdir:
        logger.error('cannot store topics without having specified an output directory')
    if doc_count is None:
        fname = 'final'
    else:
        fname = 'doc-%i' % doc_count
    fname = '%s/%s.topics' % (self.outputdir, fname)
    logger.info('saving topics to %s', fname)
    betas = self.m_lambda + self.m_eta
    np.savetxt(fname, betas)