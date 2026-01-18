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
def in_next(self, offset):
    """
        Determine whether the given offset falls within the next shard.
        This is a very small speedup: typically, we will be iterating through
        the data forward. Could save considerable time with a very large number
        of smaller shards.

        """
    if self.current_shard_n == self.n_shards:
        return False
    return self.offsets[self.current_shard_n + 1] <= offset and offset < self.offsets[self.current_shard_n + 2]