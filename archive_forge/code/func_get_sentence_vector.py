import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def get_sentence_vector(self, sentence):
    """Get a single 1-D vector representation for a given `sentence`.
        This function is workalike of the official fasttext's get_sentence_vector().

        Parameters
        ----------
        sentence : list of (str or int)
            list of words specified by string or int ids.

        Returns
        -------
        numpy.ndarray
            1-D numpy array representation of the `sentence`.

        """
    return super(FastTextKeyedVectors, self).get_mean_vector(sentence)