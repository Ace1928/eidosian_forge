import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def cg(A, b, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``. ``A`` must be a hermitian,
            positive definitive matrix with type of :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.cg`
    """
    A, M, x, b = _make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = A.shape[0]
    if maxiter is None:
        maxiter = n * 10
    if n == 0:
        return (cupy.empty_like(b), 0)
    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return (b, 0)
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
    r = b - matvec(x)
    iters = 0
    rho = 0
    while iters < maxiter:
        z = psolve(r)
        rho1 = rho
        rho = cublas.dotc(r, z)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            p = z + beta * p
        q = matvec(p)
        alpha = rho / cublas.dotc(p, q)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        resid = cublas.nrm2(r)
        if resid <= atol:
            break
    info = 0
    if iters == maxiter and (not resid <= atol):
        info = iters
    return (x, info)