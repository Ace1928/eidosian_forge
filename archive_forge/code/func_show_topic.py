import logging
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools
from gensim import interfaces, matutils, utils
from gensim.models import basemodel
from gensim.utils import is_empty
def show_topic(self, topicno, topn=10):
    """Get the words that define a topic along with their contribution.

        This is actually the left singular vector of the specified topic.

        The most important words in defining the topic (greatest absolute value) are included
        in the output, along with their contribution to the topic.

        Parameters
        ----------
        topicno : int
            The topics id number.
        topn : int
            Number of words to be included to the result.

        Returns
        -------
        list of (str, float)
            Topic representation in BoW format.

        """
    if topicno >= len(self.projection.u.T):
        return ''
    c = np.asarray(self.projection.u.T[topicno, :]).flatten()
    norm = np.sqrt(np.sum(np.dot(c, c)))
    most = matutils.argsort(np.abs(c), topn, reverse=True)
    return [(self.id2word[val], 1.0 * c[val] / norm) for val in most if val in self.id2word]