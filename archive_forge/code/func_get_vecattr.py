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
def get_vecattr(self, key, attr):
    """Get attribute value associated with given key.

        Parameters
        ----------

        key : str
            Vector key for which to fetch the attribute value.
        attr : str
            Name of the additional attribute to fetch for the given key.

        Returns
        -------

        object
            Value of the additional attribute fetched for the given key.

        """
    index = self.get_index(key)
    return self.expandos[attr][index]