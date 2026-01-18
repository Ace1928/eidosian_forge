from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def init_by_clone(self):
    """
        Initialize by copying over attributes of another ShardedCorpus
        instance saved to the output_prefix given at __init__().

        """
    temp = self.__class__.load(self.output_prefix)
    self.n_shards = temp.n_shards
    self.n_docs = temp.n_docs
    self.offsets = temp.offsets
    if temp.dim != self.dim:
        if self.dim is None:
            logger.info('Loaded dataset dimension: %d', temp.dim)
        else:
            logger.warning('Loaded dataset dimension differs from init arg dimension, using loaded dim. (loaded %d, init %d)', temp.dim, self.dim)
    self.dim = temp.dim