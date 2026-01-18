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
def rank_by_centrality(self, words, use_norm=True):
    """Rank the given words by similarity to the centroid of all the words.

        Parameters
        ----------
        words : list of str
            List of keys.
        use_norm : bool, optional
            Whether to calculate centroid using unit-normed vectors; default True.

        Returns
        -------
        list of (float, str)
            Ranked list of (similarity, key), most-similar to the centroid first.

        """
    self.fill_norms()
    used_words = [word for word in words if word in self]
    if len(used_words) != len(words):
        ignored_words = set(words) - set(used_words)
        logger.warning('vectors for words %s are not present in the model, ignoring these words', ignored_words)
    if not used_words:
        raise ValueError('cannot select a word from an empty list')
    vectors = vstack([self.get_vector(word, norm=use_norm) for word in used_words]).astype(REAL)
    mean = self.get_mean_vector(vectors, post_normalize=True)
    dists = dot(vectors, mean)
    return sorted(zip(dists, used_words), reverse=True)