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
def doesnt_match(self, words):
    """Which key from the given list doesn't go with the others?

        Parameters
        ----------
        words : list of str
            List of keys.

        Returns
        -------
        str
            The key further away from the mean of all keys.

        """
    return self.rank_by_centrality(words)[-1][1]