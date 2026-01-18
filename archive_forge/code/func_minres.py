import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
def minres(A, b, x0=None, shift=0.0, tol=1e-05, maxiter=None, M=None, callback=None, check=False):
    """Uses MINimum RESidual iteration to solve  ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        shift (int or float): If shift != 0 then the method solves
            ``(A - shift*I)x = b``
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

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.minres`
    """
    A, M, x, b = _make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 5
    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0
    xtype = x.dtype
    eps = cupy.finfo(xtype).eps
    Ax = matvec(x)
    r1 = b - Ax
    y = psolve(r1)
    beta1 = cupy.inner(r1, y)
    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return (x, 0)
    beta1 = cupy.sqrt(beta1)
    beta1 = beta1.get().item()
    if check:
        if not _check_symmetric(A, Ax, x, eps):
            raise ValueError('non-symmetric matrix')
        if not _check_symmetric(M, y, r1, eps):
            raise ValueError('non-symmetric preconditioner')
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = cupy.finfo(xtype).max
    cs = -1
    sn = 0
    w = cupy.zeros(n, dtype=xtype)
    w2 = cupy.zeros(n, dtype=xtype)
    r2 = r1
    while itn < maxiter:
        itn += 1
        s = 1.0 / beta
        v = s * y
        y = matvec(v)
        y -= shift * v
        if itn >= 2:
            y -= beta / oldb * r1
        alpha = cupy.inner(v, y)
        alpha = alpha.get().item()
        y -= alpha / beta * r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = cupy.inner(r2, y)
        beta = beta.get().item()
        beta = numpy.sqrt(beta)
        if beta < 0:
            raise ValueError('non-symmetric matrix')
        tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2
        if itn == 1:
            if beta / beta1 <= 10 * eps:
                istop = -1
        oldeps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        epsln = sn * beta
        dbar = -cs * beta
        root = numpy.linalg.norm([gbar, dbar])
        gamma = numpy.linalg.norm([gbar, beta])
        gamma = max(gamma, eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x += phi * w
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z
        Anorm = numpy.sqrt(tnorm2)
        ynorm = cupy.linalg.norm(x)
        ynorm = ynorm.get().item()
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        diag = gbar
        if diag == 0:
            diag = epsa
        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = numpy.inf
        else:
            test1 = rnorm / (Anorm * ynorm)
        if Anorm == 0:
            test2 = numpy.inf
        else:
            test2 = root / Anorm
        Acond = gmax / gmin
        if istop == 0:
            t1 = 1 + test1
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1
            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1
        if callback is not None:
            callback(x)
        if istop != 0:
            break
    if istop == 6:
        info = maxiter
    else:
        info = 0
    return (x, info)