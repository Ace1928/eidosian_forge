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
def optimal_ordering(self):
    """Performs ordering on the topics."""
    idx = matutils.argsort(self.m_lambda_sum, reverse=True)
    self.m_varphi_ss = self.m_varphi_ss[idx]
    self.m_lambda = self.m_lambda[idx, :]
    self.m_lambda_sum = self.m_lambda_sum[idx]
    self.m_Elogbeta = self.m_Elogbeta[idx, :]