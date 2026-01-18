import warnings
import numpy as np
from scipy.sparse.linalg._interface import LinearOperator
from .utils import make_system
from scipy.linalg import get_lapack_funcs
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
@_deprecate_positional_args(version='1.14')
def bicgstab(A, b, *, x0=None, tol=_NoValue, maxiter=None, M=None, callback=None, atol=0.0, rtol=1e-05):
    """Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `bicgstab` keyword argument ``tol`` is deprecated in favor of
           ``rtol`` and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import bicgstab
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_matrix(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = bicgstab(A, b, atol=1e-5)
    >>> print(exit_code)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)
    atol, _ = _get_atol_rtol('bicgstab', bnrm2, tol, atol, rtol)
    if bnrm2 == 0:
        return (postprocess(b), 0)
    n = len(b)
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot
    if maxiter is None:
        maxiter = n * 10
    matvec = A.matvec
    psolve = M.matvec
    rhotol = np.finfo(x.dtype.char).eps ** 2
    omegatol = rhotol
    rho_prev, omega, alpha, p, v = (None, None, None, None, None)
    r = b - matvec(x) if x.any() else b.copy()
    rtilde = r.copy()
    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:
            return (postprocess(x), 0)
        rho = dotprod(rtilde, r)
        if np.abs(rho) < rhotol:
            return (postprocess(x), -10)
        if iteration > 0:
            if np.abs(omega) < omegatol:
                return (postprocess(x), -11)
            beta = rho / rho_prev * (alpha / omega)
            p -= omega * v
            p *= beta
            p += r
        else:
            s = np.empty_like(r)
            p = r.copy()
        phat = psolve(p)
        v = matvec(phat)
        rv = dotprod(rtilde, v)
        if rv == 0:
            return (postprocess(x), -11)
        alpha = rho / rv
        r -= alpha * v
        s[:] = r[:]
        if np.linalg.norm(s) < atol:
            x += alpha * phat
            return (postprocess(x), 0)
        shat = psolve(s)
        t = matvec(shat)
        omega = dotprod(t, s) / dotprod(t, t)
        x += alpha * phat
        x += omega * shat
        r -= omega * t
        rho_prev = rho
        if callback:
            callback(x)
    else:
        return (postprocess(x), maxiter)