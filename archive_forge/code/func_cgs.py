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
def cgs(A, b, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None):
    """Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``.
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

    .. seealso:: :func:`scipy.sparse.linalg.cgs`
    """
    A, M, x, b = _make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = A.shape[0]
    if n == 0:
        return (cupy.empty_like(b), 0)
    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return (b, 0)
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 5
    r0 = b - matvec(x)
    rho = cupy.dot(r0, r0)
    r = r0.copy()
    u = r0
    p = r0.copy()
    iters = 0
    while True:
        y = psolve(p)
        v = matvec(y)
        sigma = cupy.dot(r0, v)
        alpha = rho / sigma
        q = u - alpha * v
        z = psolve(u + q)
        x += alpha * z
        Az = matvec(z)
        r -= alpha * Az
        r_norm = cupy.linalg.norm(r)
        iters += 1
        if callback is not None:
            callback(x)
        if r_norm <= atol or iters >= maxiter:
            break
        rho_new = cupy.dot(r0, r)
        beta = rho_new / rho
        rho = rho_new
        u = r + beta * q
        p *= beta
        p += q
        p *= beta
        p += u
    info = 0
    if iters == maxiter and (not r_norm < atol):
        info = iters
    return (x, info)