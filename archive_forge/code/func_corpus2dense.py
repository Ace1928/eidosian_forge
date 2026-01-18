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
def corpus2dense(corpus, num_terms, num_docs=None, dtype=np.float32):
    """Convert corpus into a dense numpy 2D array, with documents as columns.

    Parameters
    ----------
    corpus : iterable of iterable of (int, number)
        Input corpus in the Gensim bag-of-words format.
    num_terms : int
        Number of terms in the dictionary. X-axis of the resulting matrix.
    num_docs : int, optional
        Number of documents in the corpus. If provided, a slightly more memory-efficient code path is taken.
        Y-axis of the resulting matrix.
    dtype : data-type, optional
        Data type of the output matrix.

    Returns
    -------
    numpy.ndarray
        Dense 2D array that presents `corpus`.

    See Also
    --------
    :class:`~gensim.matutils.Dense2Corpus`
        Convert dense matrix to Gensim corpus format.

    """
    if num_docs is not None:
        docno, result = (-1, np.empty((num_terms, num_docs), dtype=dtype))
        for docno, doc in enumerate(corpus):
            result[:, docno] = sparse2full(doc, num_terms)
        assert docno + 1 == num_docs
    else:
        result = np.column_stack([sparse2full(doc, num_terms) for doc in corpus])
    return result.astype(dtype)