from itertools import product
import numpy as np
from numpy import (dot, diag, prod, logical_not, ravel, transpose,
from numpy.lib.scimath import sqrt as csqrt
from scipy.linalg import LinAlgError, bandwidth
from ._misc import norm
from ._basic import solve, inv
from ._decomp_svd import svd
from ._decomp_schur import schur, rsf2csf
from ._expm_frechet import expm_frechet, expm_cond
from ._matfuncs_sqrtm import sqrtm
from ._matfuncs_expm import pick_pade_structure, pade_UV_calc
from numpy import single  # noqa: F401
def logm(A, disp=True):
    """
    Compute matrix logarithm.

    The matrix logarithm is the inverse of
    expm: expm(logm(`A`)) == `A`

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose logarithm to evaluate
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import logm, expm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> b = logm(a)
    >>> b
    array([[-1.02571087,  2.05142174],
           [ 0.68380725,  1.02571087]])
    >>> expm(b)         # Verify expm(logm(a)) returns a
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    A = _asarray_square(A)
    import scipy.linalg._matfuncs_inv_ssq
    F = scipy.linalg._matfuncs_inv_ssq._logm(A)
    F = _maybe_real(A, F)
    errtol = 1000 * eps
    errest = norm(expm(F) - A, 1) / norm(A, 1)
    if disp:
        if not isfinite(errest) or errest >= errtol:
            print('logm result may be inaccurate, approximate err =', errest)
        return F
    else:
        return (F, errest)