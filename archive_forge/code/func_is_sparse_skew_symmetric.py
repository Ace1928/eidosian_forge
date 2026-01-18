import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def is_sparse_skew_symmetric(A) -> bool:
    """Check if a real sparse matrix A satisfies A + A.T == 0.

    Parameters
    ----------
    A : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('m must be a square matrix')
    if not isinstance(A, sp.coo_matrix):
        A = sp.coo_matrix(A)
    r, c, v = (A.row, A.col, A.data)
    tril = r >= c
    triu = c >= r
    if triu.sum() != tril.sum():
        return False
    rl = r[tril]
    cl = c[tril]
    vl = v[tril]
    ru = r[triu]
    cu = c[triu]
    vu = v[triu]
    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]
    check = np.allclose(vl + vu, 0)
    return check