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
def has_index_for(self, key):
    """Can this model return a single index for this key?

        Subclasses that synthesize vectors for out-of-vocabulary words (like
        :class:`~gensim.models.fasttext.FastText`) may respond True for a
        simple `word in wv` (`__contains__()`) check but False for this
        more-specific check.

        """
    return self.get_index(key, -1) >= 0