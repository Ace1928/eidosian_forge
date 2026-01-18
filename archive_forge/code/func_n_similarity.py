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
def n_similarity(self, ws1, ws2):
    """Compute cosine similarity between two sets of keys.

        Parameters
        ----------
        ws1 : list of str
            Sequence of keys.
        ws2: list of str
            Sequence of keys.

        Returns
        -------
        numpy.ndarray
            Similarities between `ws1` and `ws2`.

        """
    if not (len(ws1) and len(ws2)):
        raise ZeroDivisionError('At least one of the passed list is empty.')
    mean1 = self.get_mean_vector(ws1, pre_normalize=False)
    mean2 = self.get_mean_vector(ws2, pre_normalize=False)
    return dot(matutils.unitvec(mean1), matutils.unitvec(mean2))