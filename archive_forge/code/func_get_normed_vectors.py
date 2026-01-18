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
def get_normed_vectors(self):
    """Get all embedding vectors normalized to unit L2 length (euclidean), as a 2D numpy array.

        To see which key corresponds to which vector = which array row, refer
        to the :attr:`~gensim.models.keyedvectors.KeyedVectors.index_to_key` attribute.

        Returns
        -------
        numpy.ndarray:
            2D numpy array of shape ``(number_of_keys, embedding dimensionality)``, L2-normalized
            along the rows (key vectors).

        """
    self.fill_norms()
    return self.vectors / self.norms[..., np.newaxis]