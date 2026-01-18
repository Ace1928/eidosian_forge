import logging
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools
from gensim import interfaces, matutils, utils
from gensim.models import basemodel
from gensim.utils import is_empty
def stochastic_svd(corpus, rank, num_terms, chunksize=20000, extra_dims=None, power_iters=0, dtype=np.float64, eps=1e-06, random_seed=None):
    """Run truncated Singular Value Decomposition (SVD) on a sparse input.

    Parameters
    ----------
    corpus : {iterable of list of (int, float), scipy.sparse}
        Input corpus as a stream (does not have to fit in RAM)
        or a sparse matrix of shape (`num_terms`, num_documents).
    rank : int
        Desired number of factors to be retained after decomposition.
    num_terms : int
        The number of features (terms) in `corpus`.
    chunksize :  int, optional
        Number of documents to be used in each training chunk.
    extra_dims : int, optional
        Extra samples to be used besides the rank `k`. Can improve accuracy.
    power_iters: int, optional
        Number of power iteration steps to be used. Increasing the number of power iterations improves accuracy,
        but lowers performance.
    dtype : numpy.dtype, optional
        Enforces a type for elements of the decomposed matrix.
    eps: float, optional
        Percentage of the spectrum's energy to be discarded.
    random_seed: {None, int}, optional
        Random seed used to initialize the pseudo-random number generator,
         a local instance of numpy.random.RandomState instance.


    Notes
    -----
    The corpus may be larger than RAM (iterator of vectors), if `corpus` is a `scipy.sparse.csc` instead,
    it is assumed the whole corpus fits into core memory and a different (more efficient) code path is chosen.
    This may return less than the requested number of top `rank` factors, in case the input itself is of lower rank.
    The `extra_dims` (oversampling) and especially `power_iters` (power iterations) parameters affect accuracy of the
    decomposition.

    This algorithm uses `2 + power_iters` passes over the input data. In case you can only afford a single pass,
    set `onepass=True` in :class:`~gensim.models.lsimodel.LsiModel` and avoid using this function directly.

    The decomposition algorithm is based on `"Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions" <https://arxiv.org/abs/0909.4061>`_.


    Returns
    -------
    (np.ndarray 2D, np.ndarray 1D)
        The left singular vectors and the singular values of the `corpus`.

    """
    rank = int(rank)
    if extra_dims is None:
        samples = max(10, 2 * rank)
    else:
        samples = rank + int(extra_dims)
    logger.info('using %i extra samples and %i power iterations', samples - rank, power_iters)
    num_terms = int(num_terms)
    y = np.zeros(dtype=dtype, shape=(num_terms, samples))
    logger.info('1st phase: constructing %s action matrix', str(y.shape))
    random_state = np.random.RandomState(random_seed)
    if scipy.sparse.issparse(corpus):
        m, n = corpus.shape
        assert num_terms == m, f'mismatch in number of features: {m} in sparse matrix vs. {num_terms} parameter'
        o = random_state.normal(0.0, 1.0, (n, samples)).astype(y.dtype)
        sparsetools.csc_matvecs(m, n, samples, corpus.indptr, corpus.indices, corpus.data, o.ravel(), y.ravel())
        del o
        if y.dtype != dtype:
            y = y.astype(dtype)
        logger.info('orthonormalizing %s action matrix', str(y.shape))
        y = [y]
        q, _ = matutils.qr_destroy(y)
        logger.debug('running %i power iterations', power_iters)
        for _ in range(power_iters):
            q = corpus.T * q
            q = [corpus * q]
            q, _ = matutils.qr_destroy(q)
    else:
        num_docs = 0
        for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize)):
            logger.info('PROGRESS: at document #%i', chunk_no * chunksize)
            s = sum((len(doc) for doc in chunk))
            chunk = matutils.corpus2csc(chunk, num_terms=num_terms, dtype=dtype)
            m, n = chunk.shape
            assert m == num_terms
            assert n <= chunksize
            num_docs += n
            logger.debug('multiplying chunk * gauss')
            o = random_state.normal(0.0, 1.0, (n, samples)).astype(dtype)
            sparsetools.csc_matvecs(m, n, samples, chunk.indptr, chunk.indices, chunk.data, o.ravel(), y.ravel())
            del chunk, o
        y = [y]
        q, _ = matutils.qr_destroy(y)
        for power_iter in range(power_iters):
            logger.info('running power iteration #%i', power_iter + 1)
            yold = q.copy()
            q[:] = 0.0
            for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize)):
                logger.info('PROGRESS: at document #%i/%i', chunk_no * chunksize, num_docs)
                chunk = matutils.corpus2csc(chunk, num_terms=num_terms, dtype=dtype)
                tmp = chunk.T * yold
                tmp = chunk * tmp
                del chunk
                q += tmp
            del yold
            q = [q]
            q, _ = matutils.qr_destroy(q)
    qt = q[:, :samples].T.copy()
    del q
    if scipy.sparse.issparse(corpus):
        b = qt * corpus
        logger.info('2nd phase: running dense svd on %s matrix', str(b.shape))
        u, s, vt = scipy.linalg.svd(b, full_matrices=False)
        del b, vt
    else:
        x = np.zeros(shape=(qt.shape[0], qt.shape[0]), dtype=dtype)
        logger.info('2nd phase: constructing %s covariance matrix', str(x.shape))
        for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize)):
            logger.info('PROGRESS: at document #%i/%i', chunk_no * chunksize, num_docs)
            chunk = matutils.corpus2csc(chunk, num_terms=num_terms, dtype=qt.dtype)
            b = qt * chunk
            del chunk
            x += np.dot(b, b.T)
            del b
        logger.info('running dense decomposition on %s covariance matrix', str(x.shape))
        u, s, vt = scipy.linalg.svd(x)
        s = np.sqrt(s)
    q = qt.T.copy()
    del qt
    logger.info('computing the final decomposition')
    keep = clip_spectrum(s ** 2, rank, discard=eps)
    u = u[:, :keep].copy()
    s = s[:keep]
    u = np.dot(q, u)
    return (u.astype(dtype), s.astype(dtype))