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
def init_shards(self, output_prefix, corpus, shardsize=4096, dtype=_default_dtype):
    """Initialize shards from the corpus."""
    is_corpus, corpus = gensim.utils.is_corpus(corpus)
    if not is_corpus:
        raise ValueError('Cannot initialize shards without a corpus to read from! Corpus type: %s' % type(corpus))
    proposed_dim = self._guess_n_features(corpus)
    if proposed_dim != self.dim:
        if self.dim is None:
            logger.info('Deriving dataset dimension from corpus: %d', proposed_dim)
        else:
            logger.warning('Dataset dimension derived from input corpus differs from initialization argument, using corpus. (corpus %d, init arg %d)', proposed_dim, self.dim)
    self.dim = proposed_dim
    self.offsets = [0]
    start_time = time.perf_counter()
    logger.info('Running init from corpus.')
    for n, doc_chunk in enumerate(gensim.utils.grouper(corpus, chunksize=shardsize)):
        logger.info('Chunk no. %d at %f s', n, time.perf_counter() - start_time)
        current_shard = numpy.zeros((len(doc_chunk), self.dim), dtype=dtype)
        logger.debug('Current chunk dimension: %d x %d', len(doc_chunk), self.dim)
        for i, doc in enumerate(doc_chunk):
            doc = dict(doc)
            current_shard[i][list(doc)] = list(doc.values())
        if self.sparse_serialization:
            current_shard = sparse.csr_matrix(current_shard)
        self.save_shard(current_shard)
    end_time = time.perf_counter()
    logger.info('Built %d shards in %f s.', self.n_shards, end_time - start_time)