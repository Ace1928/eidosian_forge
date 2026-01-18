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
def signm(A, disp=True):
    """
    Matrix sign function.

    Extension of the scalar sign(x) to matrices.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix at which to evaluate the sign function
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    signm : (N, N) ndarray
        Value of the sign function at `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    Examples
    --------
    >>> from scipy.linalg import signm, eigvals
    >>> a = [[1,2,3], [1,2,1], [1,1,1]]
    >>> eigvals(a)
    array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])
    >>> eigvals(signm(a))
    array([-1.+0.j,  1.+0.j,  1.+0.j])

    """
    A = _asarray_square(A)

    def rounded_sign(x):
        rx = np.real(x)
        if rx.dtype.char == 'f':
            c = 1000.0 * feps * amax(x)
        else:
            c = 1000.0 * eps * amax(x)
        return sign((absolute(rx) > c) * rx)
    result, errest = funm(A, rounded_sign, disp=0)
    errtol = {0: 1000.0 * feps, 1: 1000.0 * eps}[_array_precision[result.dtype.char]]
    if errest < errtol:
        return result
    vals = svd(A, compute_uv=False)
    max_sv = np.amax(vals)
    c = 0.5 / max_sv
    S0 = A + c * np.identity(A.shape[0])
    prev_errest = errest
    for i in range(100):
        iS0 = inv(S0)
        S0 = 0.5 * (S0 + iS0)
        Pp = 0.5 * (dot(S0, S0) + S0)
        errest = norm(dot(Pp, Pp) - Pp, 1)
        if errest < errtol or prev_errest == errest:
            break
        prev_errest = errest
    if disp:
        if not isfinite(errest) or errest >= errtol:
            print('signm result may be inaccurate, approximate err =', errest)
        return S0
    else:
        return (S0, errest)