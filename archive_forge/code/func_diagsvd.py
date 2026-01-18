import numpy
from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs, _compute_lwork
from ._decomp import _asarray_validated
def diagsvd(s, M, N):
    """
    Construct the sigma matrix in SVD from singular values and size M, N.

    Parameters
    ----------
    s : (M,) or (N,) array_like
        Singular values
    M : int
        Size of the matrix whose singular values are `s`.
    N : int
        Size of the matrix whose singular values are `s`.

    Returns
    -------
    S : (M, N) ndarray
        The S-matrix in the singular value decomposition

    See Also
    --------
    svd : Singular value decomposition of a matrix
    svdvals : Compute singular values of a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import diagsvd
    >>> vals = np.array([1, 2, 3])  # The array representing the computed svd
    >>> diagsvd(vals, 3, 4)
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0]])
    >>> diagsvd(vals, 4, 3)
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3],
           [0, 0, 0]])

    """
    part = diag(s)
    typ = part.dtype.char
    MorN = len(s)
    if MorN == M:
        return numpy.hstack((part, zeros((M, N - M), dtype=typ)))
    elif MorN == N:
        return r_[part, zeros((M - N, N), dtype=typ)]
    else:
        raise ValueError('Length of s must be M or N.')