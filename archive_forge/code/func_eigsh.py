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
def eigsh(a, k=6, *, which='LM', ncv=None, maxiter=None, tol=0, return_eigenvectors=True):
    """
    Find ``k`` eigenvalues and eigenvectors of the real symmetric square
    matrix or complex Hermitian matrix ``A``.

    Solves ``Ax = wx``, the standard eigenvalue problem for ``w`` eigenvalues
    with corresponding eigenvectors ``x``.

    Args:
        a (ndarray, spmatrix or LinearOperator): A symmetric square matrix with
            dimension ``(n, n)``. ``a`` must :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        k (int): The number of eigenvalues and eigenvectors to compute. Must be
            ``1 <= k < n``.
        which (str): 'LM' or 'LA'. 'LM': finds ``k`` largest (in magnitude)
            eigenvalues. 'LA': finds ``k`` largest (algebraic) eigenvalues.
            'SA': finds ``k`` smallest (algebraic) eigenvalues.

        ncv (int): The number of Lanczos vectors generated. Must be
            ``k + 1 < ncv < n``. If ``None``, default value is used.
        maxiter (int): Maximum number of Lanczos update iterations.
            If ``None``, default value is used.
        tol (float): Tolerance for residuals ``||Ax - wx||``. If ``0``, machine
            precision is used.
        return_eigenvectors (bool): If ``True``, returns eigenvectors in
            addition to eigenvalues.

    Returns:
        tuple:
            If ``return_eigenvectors is True``, it returns ``w`` and ``x``
            where ``w`` is eigenvalues and ``x`` is eigenvectors. Otherwise,
            it returns only ``w``.

    .. seealso:: :func:`scipy.sparse.linalg.eigsh`

    .. note::
        This function uses the thick-restart Lanczos methods
        (https://sdm.lbl.gov/~kewu/ps/trlan.html).

    """
    n = a.shape[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('expected square matrix (shape: {})'.format(a.shape))
    if a.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(a.dtype))
    if k <= 0:
        raise ValueError('k must be greater than 0 (actual: {})'.format(k))
    if k >= n:
        raise ValueError('k must be smaller than n (actual: {})'.format(k))
    if which not in ('LM', 'LA', 'SA'):
        raise ValueError("which must be 'LM','LA'or'SA' (actual: {})".format(which))
    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = numpy.finfo(a.dtype).eps
    alpha = cupy.zeros((ncv,), dtype=a.dtype)
    beta = cupy.zeros((ncv,), dtype=a.dtype.char.lower())
    V = cupy.empty((ncv, n), dtype=a.dtype)
    u = cupy.random.random((n,)).astype(a.dtype)
    V[0] = u / cublas.nrm2(u)
    upadte_impl = 'fast'
    if upadte_impl == 'fast':
        lanczos = _lanczos_fast(a, n, ncv)
    else:
        lanczos = _lanczos_asis
    lanczos(a, V, u, alpha, beta, 0, ncv)
    iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s
    beta_k = beta[-1] * s[-1, :]
    res = cublas.nrm2(beta_k)
    uu = cupy.empty((k,), dtype=a.dtype)
    while res > tol and iter < maxiter:
        beta[:k] = 0
        alpha[:k] = w
        V[:k] = x.T
        cublas.gemv(_cublas.CUBLAS_OP_C, 1, V[:k].T, u, 0, uu)
        cublas.gemv(_cublas.CUBLAS_OP_N, -1, V[:k].T, uu, 1, u)
        V[k] = u / cublas.nrm2(u)
        u[...] = a @ V[k]
        cublas.dotc(V[k], u, out=alpha[k])
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        cublas.nrm2(u, out=beta[k])
        V[k + 1] = u / beta[k]
        lanczos(a, V, u, alpha, beta, k + 1, ncv)
        iter += ncv - k
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s
        beta_k = beta[-1] * s[-1, :]
        res = cublas.nrm2(beta_k)
    if return_eigenvectors:
        idx = cupy.argsort(w)
        return (w[idx], x[:, idx])
    else:
        return cupy.sort(w)