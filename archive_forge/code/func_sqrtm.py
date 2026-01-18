import numpy as np
from scipy._lib._util import _asarray_validated
from ._misc import norm
from .lapack import ztrsyl, dtrsyl
from ._decomp_schur import schur, rsf2csf
from ._matfuncs_sqrtm_triu import within_block_loop  # noqa: E402
def sqrtm(A, disp=True, blocksize=64):
    """
    Matrix square root.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose square root to evaluate
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)
    blocksize : integer, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `A`. The dtype is float or complex.
        The precision (data size) is determined based on the precision of
        input `A`. When the dtype is float, the precision is the same as `A`.
        When the dtype is complex, the precision is double that of `A`. The
        precision might be clipped by each dtype precision range.

    errest : float
        (if disp == False)

        Frobenius norm of the estimated error, ||err||_F / ||A||_F

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import sqrtm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> r = sqrtm(a)
    >>> r
    array([[ 0.75592895,  1.13389342],
           [ 0.37796447,  1.88982237]])
    >>> r.dot(r)
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    byte_size = np.asarray(A).dtype.itemsize
    A = _asarray_validated(A, check_finite=True, as_inexact=True)
    if len(A.shape) != 2:
        raise ValueError('Non-matrix input to matrix function.')
    if blocksize < 1:
        raise ValueError('The blocksize should be at least 1.')
    keep_it_real = np.isrealobj(A)
    if keep_it_real:
        T, Z = schur(A)
        if not np.allclose(T, np.triu(T)):
            T, Z = rsf2csf(T, Z)
    else:
        T, Z = schur(A, output='complex')
    failflag = False
    try:
        R = _sqrtm_triu(T, blocksize=blocksize)
        ZH = np.conjugate(Z).T
        X = Z.dot(R).dot(ZH)
        if not np.iscomplexobj(X):
            X = X.astype(f'f{np.clip(byte_size, 2, 16)}', copy=False)
        elif hasattr(np, 'complex256'):
            X = X.astype(f'c{np.clip(byte_size * 2, 8, 32)}', copy=False)
        else:
            X = X.astype(f'c{np.clip(byte_size * 2, 8, 16)}', copy=False)
    except SqrtmError:
        failflag = True
        X = np.empty_like(A)
        X.fill(np.nan)
    if disp:
        if failflag:
            print('Failed to find a square root.')
        return X
    else:
        try:
            arg2 = norm(X.dot(X) - A, 'fro') ** 2 / norm(A, 'fro')
        except ValueError:
            arg2 = np.inf
        return (X, arg2)