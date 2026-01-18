import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def svds(a, k=6, *, ncv=None, tol=0, which='LM', maxiter=None, return_singular_vectors=True):
    """Finds the largest ``k`` singular values/vectors for a sparse matrix.

    Args:
        a (ndarray, spmatrix or LinearOperator): A real or complex array with
            dimension ``(m, n)``. ``a`` must :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        k (int): The number of singular values/vectors to compute. Must be
            ``1 <= k < min(m, n)``.
        ncv (int): The number of Lanczos vectors generated. Must be
            ``k + 1 < ncv < min(m, n)``. If ``None``, default value is used.
        tol (float): Tolerance for singular values. If ``0``, machine precision
            is used.
        which (str): Only 'LM' is supported. 'LM': finds ``k`` largest singular
            values.
        maxiter (int): Maximum number of Lanczos update iterations.
            If ``None``, default value is used.
        return_singular_vectors (bool): If ``True``, returns singular vectors
            in addition to singular values.

    Returns:
        tuple:
            If ``return_singular_vectors`` is ``True``, it returns ``u``, ``s``
            and ``vt`` where ``u`` is left singular vectors, ``s`` is singular
            values and ``vt`` is right singular vectors. Otherwise, it returns
            only ``s``.

    .. seealso:: :func:`scipy.sparse.linalg.svds`

    .. note::
        This is a naive implementation using cupyx.scipy.sparse.linalg.eigsh as
        an eigensolver on ``a.H @ a`` or ``a @ a.H``.

    """
    if a.ndim != 2:
        raise ValueError('expected 2D (shape: {})'.format(a.shape))
    if a.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(a.dtype))
    m, n = a.shape
    if k <= 0:
        raise ValueError('k must be greater than 0 (actual: {})'.format(k))
    if k >= min(m, n):
        raise ValueError('k must be smaller than min(m, n) (actual: {})'.format(k))
    a = _interface.aslinearoperator(a)
    if m >= n:
        aH, a = (a.H, a)
    else:
        aH, a = (a, a.H)
    if return_singular_vectors:
        w, x = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=True)
    else:
        w = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=False)
    w = cupy.maximum(w, 0)
    t = w.dtype.char.lower()
    factor = {'f': 1000.0, 'd': 1000000.0}
    cond = factor[t] * numpy.finfo(t).eps
    cutoff = cond * cupy.max(w)
    above_cutoff = w > cutoff
    n_large = above_cutoff.sum().item()
    s = cupy.zeros_like(w)
    s[:n_large] = cupy.sqrt(w[above_cutoff])
    if not return_singular_vectors:
        return s
    x = x[:, above_cutoff]
    if m >= n:
        v = x
        u = a @ v / s[:n_large]
    else:
        u = x
        v = a @ u / s[:n_large]
    u = _augmented_orthnormal_cols(u, k - n_large)
    v = _augmented_orthnormal_cols(v, k - n_large)
    return (u, s, v.conj().T)