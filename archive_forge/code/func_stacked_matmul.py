from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def stacked_matmul(a, b):
    """Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).

    Empirical optimization. Use outer Python loop and BLAS for large
    matrices, otherwise use a single einsum call.
    """
    if a.shape[1] > 50:
        out = np.empty((a.shape[0], a.shape[1], b.shape[2]))
        for i in range(a.shape[0]):
            out[i] = np.dot(a[i], b[i])
        return out
    else:
        return np.einsum('...ij,...jk->...ik', a, b)