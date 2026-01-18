import warnings
import numpy as np
from scipy.sparse.linalg._interface import LinearOperator
from .utils import make_system
from scipy.linalg import get_lapack_funcs
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
@_deprecate_positional_args(version='1.14')
def qmr(A, b, x0=None, *, tol=_NoValue, maxiter=None, M1=None, M2=None, callback=None, atol=0.0, rtol=1e-05):
    """Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    atol, rtol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M1 : {sparse matrix, ndarray, LinearOperator}
        Left preconditioner for A.
    M2 : {sparse matrix, ndarray, LinearOperator}
        Right preconditioner for A. Used together with the left
        preconditioner M1.  The matrix M1@A@M2 should have better
        conditioned than A alone.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `qmr` keyword argument ``tol`` is deprecated in favor of ``rtol``
           and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    See Also
    --------
    LinearOperator

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import qmr
    >>> A = csc_matrix([[3., 2., 0.], [1., -1., 0.], [0., 5., 1.]])
    >>> b = np.array([2., 4., -1.])
    >>> x, exitCode = qmr(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A_ = A
    A, M, x, b, postprocess = make_system(A, None, x0, b)
    bnrm2 = np.linalg.norm(b)
    atol, _ = _get_atol_rtol('qmr', bnrm2, tol, atol, rtol)
    if bnrm2 == 0:
        return (postprocess(b), 0)
    if M1 is None and M2 is None:
        if hasattr(A_, 'psolve'):

            def left_psolve(b):
                return A_.psolve(b, 'left')

            def right_psolve(b):
                return A_.psolve(b, 'right')

            def left_rpsolve(b):
                return A_.rpsolve(b, 'left')

            def right_rpsolve(b):
                return A_.rpsolve(b, 'right')
            M1 = LinearOperator(A.shape, matvec=left_psolve, rmatvec=left_rpsolve)
            M2 = LinearOperator(A.shape, matvec=right_psolve, rmatvec=right_rpsolve)
        else:

            def id(b):
                return b
            M1 = LinearOperator(A.shape, matvec=id, rmatvec=id)
            M2 = LinearOperator(A.shape, matvec=id, rmatvec=id)
    n = len(b)
    if maxiter is None:
        maxiter = n * 10
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot
    rhotol = np.finfo(x.dtype.char).eps
    betatol = rhotol
    gammatol = rhotol
    deltatol = rhotol
    epsilontol = rhotol
    xitol = rhotol
    r = b - A.matvec(x) if x.any() else b.copy()
    vtilde = r.copy()
    y = M1.matvec(vtilde)
    rho = np.linalg.norm(y)
    wtilde = r.copy()
    z = M2.rmatvec(wtilde)
    xi = np.linalg.norm(z)
    gamma, eta, theta = (1, -1, 0)
    v = np.empty_like(vtilde)
    w = np.empty_like(wtilde)
    epsilon, q, d, p, s = (None, None, None, None, None)
    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:
            return (postprocess(x), 0)
        if np.abs(rho) < rhotol:
            return (postprocess(x), -10)
        if np.abs(xi) < xitol:
            return (postprocess(x), -15)
        v[:] = vtilde[:]
        v *= 1 / rho
        y *= 1 / rho
        w[:] = wtilde[:]
        w *= 1 / xi
        z *= 1 / xi
        delta = dotprod(z, y)
        if np.abs(delta) < deltatol:
            return (postprocess(x), -13)
        ytilde = M2.matvec(y)
        ztilde = M1.rmatvec(z)
        if iteration > 0:
            ytilde -= xi * delta / epsilon * p
            p[:] = ytilde[:]
            ztilde -= rho * (delta / epsilon).conj() * q
            q[:] = ztilde[:]
        else:
            p = ytilde.copy()
            q = ztilde.copy()
        ptilde = A.matvec(p)
        epsilon = dotprod(q, ptilde)
        if np.abs(epsilon) < epsilontol:
            return (postprocess(x), -14)
        beta = epsilon / delta
        if np.abs(beta) < betatol:
            return (postprocess(x), -11)
        vtilde[:] = ptilde[:]
        vtilde -= beta * v
        y = M1.matvec(vtilde)
        rho_prev = rho
        rho = np.linalg.norm(y)
        wtilde[:] = w[:]
        wtilde *= -beta.conj()
        wtilde += A.rmatvec(q)
        z = M2.rmatvec(wtilde)
        xi = np.linalg.norm(z)
        gamma_prev = gamma
        theta_prev = theta
        theta = rho / (gamma_prev * np.abs(beta))
        gamma = 1 / np.sqrt(1 + theta ** 2)
        if np.abs(gamma) < gammatol:
            return (postprocess(x), -12)
        eta *= -(rho_prev / beta) * (gamma / gamma_prev) ** 2
        if iteration > 0:
            d *= (theta_prev * gamma) ** 2
            d += eta * p
            s *= (theta_prev * gamma) ** 2
            s += eta * ptilde
        else:
            d = p.copy()
            d *= eta
            s = ptilde.copy()
            s *= eta
        x += d
        r -= s
        if callback:
            callback(x)
    else:
        return (postprocess(x), maxiter)