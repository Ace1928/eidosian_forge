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
def get_mean_vector(self, keys, weights=None, pre_normalize=True, post_normalize=False, ignore_missing=True):
    """Get the mean vector for a given list of keys.

        Parameters
        ----------

        keys : list of (str or int or ndarray)
            Keys specified by string or int ids or numpy array.
        weights : list of float or numpy.ndarray, optional
            1D array of same size of `keys` specifying the weight for each key.
        pre_normalize : bool, optional
            Flag indicating whether to normalize each keyvector before taking mean.
            If False, individual keyvector will not be normalized.
        post_normalize: bool, optional
            Flag indicating whether to normalize the final mean vector.
            If True, normalized mean vector will be return.
        ignore_missing : bool, optional
            If False, will raise error if a key doesn't exist in vocabulary.

        Returns
        -------

        numpy.ndarray
            Mean vector for the list of keys.

        Raises
        ------

        ValueError
            If the size of the list of `keys` and `weights` doesn't match.
        KeyError
            If any of the key doesn't exist in vocabulary and `ignore_missing` is false.

        """
    if len(keys) == 0:
        raise ValueError('cannot compute mean with no input')
    if isinstance(weights, list):
        weights = np.array(weights)
    if weights is None:
        weights = np.ones(len(keys))
    if len(keys) != weights.shape[0]:
        raise ValueError('keys and weights array must have same number of elements')
    mean = np.zeros(self.vector_size, self.vectors.dtype)
    total_weight = 0
    for idx, key in enumerate(keys):
        if isinstance(key, ndarray):
            mean += weights[idx] * key
            total_weight += abs(weights[idx])
        elif self.__contains__(key):
            vec = self.get_vector(key, norm=pre_normalize)
            mean += weights[idx] * vec
            total_weight += abs(weights[idx])
        elif not ignore_missing:
            raise KeyError(f"Key '{key}' not present in vocabulary")
    if total_weight > 0:
        mean = mean / total_weight
    if post_normalize:
        mean = matutils.unitvec(mean).astype(REAL)
    return mean