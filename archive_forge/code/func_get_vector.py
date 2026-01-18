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
def get_vector(self, key, norm=False):
    """Get the key's vector, as a 1D numpy array.

        Parameters
        ----------

        key : str
            Key for vector to return.
        norm : bool, optional
            If True, the resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------

        numpy.ndarray
            Vector for the specified key.

        Raises
        ------

        KeyError
            If the given key doesn't exist.

        """
    index = self.get_index(key)
    if norm:
        self.fill_norms()
        result = self.vectors[index] / self.norms[index]
    else:
        result = self.vectors[index]
    result.setflags(write=False)
    return result