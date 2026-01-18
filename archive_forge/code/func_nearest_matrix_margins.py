import warnings
import numpy as np
from scipy import interpolate, stats
def nearest_matrix_margins(mat, maxiter=100, tol=1e-08):
    """nearest matrix with uniform margins

    Parameters
    ----------
    mat : array_like, 2-D
        Matrix that will be converted to have uniform margins.
        Currently, `mat` has to be two dimensional.
    maxiter : in
        Maximum number of iterations.
    tol : float
        Tolerance for convergence, defined for difference between largest and
        smallest margin in each dimension.

    Returns
    -------
    ndarray, nearest matrix with uniform margins.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.

    changed in 0.14 to support k_dim > 2.


    """
    pc = np.asarray(mat)
    converged = False
    for _ in range(maxiter):
        pc0 = pc.copy()
        for ax in range(pc.ndim):
            axs = tuple([i for i in range(pc.ndim) if not i == ax])
            pc0 /= pc.sum(axis=axs, keepdims=True)
        pc = pc0
        pc /= pc.sum()
        mptps = []
        for ax in range(pc.ndim):
            axs = tuple([i for i in range(pc.ndim) if not i == ax])
            marg = pc.sum(axis=axs, keepdims=False)
            mptps.append(np.ptp(marg))
        if max(mptps) < tol:
            converged = True
            break
    if not converged:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.warn('Iterations did not converge, maxiter reached', ConvergenceWarning)
    return pc