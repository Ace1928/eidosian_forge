import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def reopen_shard(self):
    """Reopen an incomplete shard."""
    assert self.shards
    if self.fresh_docs:
        raise ValueError('cannot reopen a shard with fresh documents in index')
    last_shard = self.shards[-1]
    last_index = last_shard.get_index()
    logger.info('reopening an incomplete shard of %i documents', len(last_shard))
    self.fresh_docs = list(last_index.index)
    self.fresh_nnz = last_shard.num_nnz
    del self.shards[-1]
    logger.debug('reopen complete')