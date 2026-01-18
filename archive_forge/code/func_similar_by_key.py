import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def similar_by_key(self, key, topn=10, restrict_vocab=None):
    """Find the top-N most similar keys.

        Parameters
        ----------
        key : str
            Key
        topn : int or None, optional
            Number of top-N similar keys to return. If topn is None, similar_by_key returns
            the vector of similarity scores.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 key vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (key, similarity) is returned.
            When `topn` is None, then similarities for all keys are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
    return self.most_similar(positive=[key], topn=topn, restrict_vocab=restrict_vocab)