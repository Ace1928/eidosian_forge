import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def query_shard(args):
    """Helper for request query from shard, same as shard[query].

    Parameters
    ---------
    args : (list of (int, number), :class:`~gensim.interfaces.SimilarityABC`)
        Query and Shard instances

    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        Similarities of the query against documents indexed in this shard.

    """
    query, shard = args
    logger.debug('querying shard %s num_best=%s in process %s', shard, shard.num_best, os.getpid())
    result = shard[query]
    logger.debug('finished querying shard %s in process %s', shard, os.getpid())
    return result