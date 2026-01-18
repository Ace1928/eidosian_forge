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
def load_shard(self, n):
    """
        Load (unpickle) the n-th shard as the "live" part of the dataset
        into the Dataset object."""
    if self.current_shard_n == n:
        return
    filename = self._shard_name(n)
    if not os.path.isfile(filename):
        raise ValueError('Attempting to load nonexistent shard no. %s' % n)
    shard = gensim.utils.unpickle(filename)
    self.current_shard = shard
    self.current_shard_n = n
    self.current_offset = self.offsets[n]