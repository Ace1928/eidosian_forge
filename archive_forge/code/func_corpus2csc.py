from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
    """Convert a streamed corpus in bag-of-words format into a sparse matrix `scipy.sparse.csc_matrix`,
    with documents as columns.

    Notes
    -----
    If the number of terms, documents and non-zero elements is known, you can pass
    them here as parameters and a (much) more memory efficient code path will be taken.

    Parameters
    ----------
    corpus : iterable of iterable of (int, number)
        Input corpus in BoW format
    num_terms : int, optional
        Number of terms in `corpus`. If provided, the `corpus.num_terms` attribute (if any) will be ignored.
    dtype : data-type, optional
        Data type of output CSC matrix.
    num_docs : int, optional
        Number of documents in `corpus`. If provided, the `corpus.num_docs` attribute (in any) will be ignored.
    num_nnz : int, optional
        Number of non-zero elements in `corpus`. If provided, the `corpus.num_nnz` attribute (if any) will be ignored.
    printprogress : int, optional
        Log a progress message at INFO level once every `printprogress` documents. 0 to turn off progress logging.

    Returns
    -------
    scipy.sparse.csc_matrix
        `corpus` converted into a sparse CSC matrix.

    See Also
    --------
    :class:`~gensim.matutils.Sparse2Corpus`
        Convert sparse format to Gensim corpus format.

    """
    try:
        if num_terms is None:
            num_terms = corpus.num_terms
        if num_docs is None:
            num_docs = corpus.num_docs
        if num_nnz is None:
            num_nnz = corpus.num_nnz
    except AttributeError:
        pass
    if printprogress:
        logger.info('creating sparse matrix from corpus')
    if num_terms is not None and num_docs is not None and (num_nnz is not None):
        posnow, indptr = (0, [0])
        indices = np.empty((num_nnz,), dtype=np.int32)
        data = np.empty((num_nnz,), dtype=dtype)
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info('PROGRESS: at document #%i/%i', docno, num_docs)
            posnext = posnow + len(doc)
            indices[posnow:posnext], data[posnow:posnext] = zip(*doc) if doc else ([], [])
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, 'mismatch between supplied and computed number of non-zeros'
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    else:
        num_nnz, data, indices, indptr = (0, [], [], [0])
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info('PROGRESS: at document #%i', docno)
            doc_indices, doc_data = zip(*doc) if doc else ([], [])
            indices.extend(doc_indices)
            data.extend(doc_data)
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_terms is None:
            num_terms = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        data = np.asarray(data, dtype=dtype)
        indices = np.asarray(indices)
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    return result