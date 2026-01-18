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
def scipy2scipy_clipped(matrix, topn, eps=1e-09):
    """Get the 'topn' elements of the greatest magnitude (absolute value) from a `scipy.sparse` vector or matrix.

    Parameters
    ----------
    matrix : `scipy.sparse`
        Input vector or matrix (1D or 2D sparse array).
    topn : int
        Number of greatest elements, in absolute value, to return.
    eps : float
        Ignored.

    Returns
    -------
    `scipy.sparse.csr.csr_matrix`
        Clipped matrix.

    """
    if not scipy.sparse.issparse(matrix):
        raise ValueError("'%s' is not a scipy sparse vector." % matrix)
    if topn <= 0:
        return scipy.sparse.csr_matrix([])
    if matrix.shape[0] == 1:
        biggest = argsort(abs(matrix.data), topn, reverse=True)
        indices, data = (matrix.indices.take(biggest), matrix.data.take(biggest))
        return scipy.sparse.csr_matrix((data, indices, [0, len(indices)]))
    else:
        matrix_indices = []
        matrix_data = []
        matrix_indptr = [0]
        matrix_abs = abs(matrix)
        for i in range(matrix.shape[0]):
            v = matrix.getrow(i)
            v_abs = matrix_abs.getrow(i)
            biggest = argsort(v_abs.data, topn, reverse=True)
            indices, data = (v.indices.take(biggest), v.data.take(biggest))
            matrix_data.append(data)
            matrix_indices.append(indices)
            matrix_indptr.append(matrix_indptr[-1] + min(len(indices), topn))
        matrix_indices = np.concatenate(matrix_indices).ravel()
        matrix_data = np.concatenate(matrix_data).ravel()
        return scipy.sparse.csr.csr_matrix((matrix_data, matrix_indices, matrix_indptr), shape=(matrix.shape[0], np.max(matrix_indices) + 1))