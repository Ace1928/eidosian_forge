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
def show_topic_terms(self, topic_data, num_words):
    """Give the topic terms along with their probabilities for a single topic data.

        Parameters
        ----------
        topic_data : list of (str, numpy.float)
            Contains probabilities for each word id belonging to a single topic.
        num_words : int
            Number of words for which probabilities are to be extracted from the given single topic data.

        Returns
        -------
        list of (str, numpy.float)
            A sequence of topic terms and their probabilities.

        """
    return [(self.dictionary[wid], weight) for weight, wid in topic_data[:num_words]]