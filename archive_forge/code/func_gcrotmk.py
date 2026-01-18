import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
from .iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
@_deprecate_positional_args(version='1.14.0')
def gcrotmk(A, b, x0=None, *, tol=_NoValue, maxiter=1000, M=None, callback=None, m=20, k=None, CU=None, discard_C=False, truncate='oldest', atol=None, rtol=1e-05):
    """
    Solve a matrix equation using flexible GCROT(m,k) algorithm.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5``, the default for ``atol`` is ``rtol``.

        .. warning::

           The default value for ``atol`` will be changed to ``0.0`` in
           SciPy 1.14.0.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner
        can vary from iteration to iteration. Effective preconditioning
        dramatically improves the rate of convergence, which implies that
        fewer iterations are needed to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    m : int, optional
        Number of inner FGMRES iterations per each outer iteration.
        Default: 20
    k : int, optional
        Number of vectors to carry between inner FGMRES iterations.
        According to [2]_, good values are around m.
        Default: m
    CU : list of tuples, optional
        List of tuples ``(c, u)`` which contain the columns of the matrices
        C and U in the GCROT(m,k) algorithm. For details, see [2]_.
        The list given and vectors contained in it are modified in-place.
        If not given, start from empty matrices. The ``c`` elements in the
        tuples can be ``None``, in which case the vectors are recomputed
        via ``c = A u`` on start and orthogonalized as described in [3]_.
    discard_C : bool, optional
        Discard the C-vectors at the end. Useful if recycling Krylov subspaces
        for different linear systems.
    truncate : {'oldest', 'smallest'}, optional
        Truncation scheme to use. Drop: oldest vectors, or vectors with
        smallest singular values using the scheme discussed in [1,2].
        See [2]_ for detailed comparison.
        Default: 'oldest'
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `gcrotmk` keyword argument ``tol`` is deprecated in favor of
           ``rtol`` and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The solution found.
    info : int
        Provides convergence information:

        * 0  : successful exit
        * >0 : convergence to tolerance not achieved, number of iterations

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import gcrotmk
    >>> R = np.random.randn(5, 5)
    >>> A = csc_matrix(R)
    >>> b = np.random.randn(5)
    >>> x, exit_code = gcrotmk(A, b, atol=1e-5)
    >>> print(exit_code)
    0
    >>> np.allclose(A.dot(x), b)
    True

    References
    ----------
    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
           methods'', SIAM J. Numer. Anal. 36, 864 (1999).
    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
           of GCROT for solving nonsymmetric linear systems'',
           SIAM J. Sci. Comput. 32, 172 (2010).
    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
           ''Recycling Krylov subspaces for sequences of linear systems'',
           SIAM J. Sci. Comput. 28, 1651 (2006).

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    if not np.isfinite(b).all():
        raise ValueError('RHS must contain only finite numbers')
    if truncate not in ('oldest', 'smallest'):
        raise ValueError(f"Invalid value for 'truncate': {truncate!r}")
    matvec = A.matvec
    psolve = M.matvec
    if CU is None:
        CU = []
    if k is None:
        k = m
    axpy, dot, scal = (None, None, None)
    if x0 is None:
        r = b.copy()
    else:
        r = b - matvec(x)
    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r))
    b_norm = nrm2(b)
    atol, rtol = _get_atol_rtol('gcrotmk', b_norm, tol, atol, rtol)
    if b_norm == 0:
        x = b
        return (postprocess(x), 0)
    if discard_C:
        CU[:] = [(None, u) for c, u in CU]
    if CU:
        CU.sort(key=lambda cu: cu[0] is not None)
        C = np.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
        us = []
        j = 0
        while CU:
            c, u = CU.pop(0)
            if c is None:
                c = matvec(u)
            C[:, j] = c
            j += 1
            us.append(u)
        Q, R, P = qr(C, overwrite_a=True, mode='economic', pivoting=True)
        del C
        cs = list(Q.T)
        new_us = []
        for j in range(len(cs)):
            u = us[P[j]]
            for i in range(j):
                u = axpy(us[P[i]], u, u.shape[0], -R[i, j])
            if abs(R[j, j]) < 1e-12 * abs(R[0, 0]):
                break
            u = scal(1.0 / R[j, j], u)
            new_us.append(u)
        CU[:] = list(zip(cs, new_us))[::-1]
    if CU:
        axpy, dot = get_blas_funcs(['axpy', 'dot'], (r,))
        for c, u in CU:
            yc = dot(c, r)
            x = axpy(u, x, x.shape[0], yc)
            r = axpy(c, r, r.shape[0], -yc)
    for j_outer in range(maxiter):
        if callback is not None:
            callback(x)
        beta = nrm2(r)
        beta_tol = max(atol, rtol * b_norm)
        if beta <= beta_tol and (j_outer > 0 or CU):
            r = b - matvec(x)
            beta = nrm2(r)
        if beta <= beta_tol:
            j_outer = -1
            break
        ml = m + max(k - len(CU), 0)
        cs = [c for c, u in CU]
        try:
            Q, R, B, vs, zs, y, pres = _fgmres(matvec, r / beta, ml, rpsolve=psolve, atol=max(atol, rtol * b_norm) / beta, cs=cs)
            y *= beta
        except LinAlgError:
            break
        ux = zs[0] * y[0]
        for z, yc in zip(zs[1:], y[1:]):
            ux = axpy(z, ux, ux.shape[0], yc)
        by = B.dot(y)
        for cu, byc in zip(CU, by):
            c, u = cu
            ux = axpy(u, ux, ux.shape[0], -byc)
        hy = Q.dot(R.dot(y))
        cx = vs[0] * hy[0]
        for v, hyc in zip(vs[1:], hy[1:]):
            cx = axpy(v, cx, cx.shape[0], hyc)
        try:
            alpha = 1 / nrm2(cx)
            if not np.isfinite(alpha):
                raise FloatingPointError()
        except (FloatingPointError, ZeroDivisionError):
            continue
        cx = scal(alpha, cx)
        ux = scal(alpha, ux)
        gamma = dot(cx, r)
        r = axpy(cx, r, r.shape[0], -gamma)
        x = axpy(ux, x, x.shape[0], gamma)
        if truncate == 'oldest':
            while len(CU) >= k and CU:
                del CU[0]
        elif truncate == 'smallest':
            if len(CU) >= k and CU:
                D = solve(R[:-1, :].T, B.T).T
                W, sigma, V = svd(D)
                new_CU = []
                for j, w in enumerate(W[:, :k - 1].T):
                    c, u = CU[0]
                    c = c * w[0]
                    u = u * w[0]
                    for cup, wp in zip(CU[1:], w[1:]):
                        cp, up = cup
                        c = axpy(cp, c, c.shape[0], wp)
                        u = axpy(up, u, u.shape[0], wp)
                    for cp, up in new_CU:
                        alpha = dot(cp, c)
                        c = axpy(cp, c, c.shape[0], -alpha)
                        u = axpy(up, u, u.shape[0], -alpha)
                    alpha = nrm2(c)
                    c = scal(1.0 / alpha, c)
                    u = scal(1.0 / alpha, u)
                    new_CU.append((c, u))
                CU[:] = new_CU
        CU.append((cx, ux))
    CU.append((None, x.copy()))
    if discard_C:
        CU[:] = [(None, uz) for cz, uz in CU]
    return (postprocess(x), j_outer + 1)