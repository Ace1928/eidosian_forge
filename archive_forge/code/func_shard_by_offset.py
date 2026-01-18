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
def shard_by_offset(self, offset):
    """
        Determine which shard the given offset belongs to. If the offset
        is greater than the number of available documents, raises a
        `ValueError`.

        Assumes that all shards have the same size.

        """
    k = int(offset / self.shardsize)
    if offset >= self.n_docs:
        raise ValueError('Too high offset specified (%s), available docs: %s' % (offset, self.n_docs))
    if offset < 0:
        raise ValueError('Negative offset %s currently not supported.' % offset)
    return k