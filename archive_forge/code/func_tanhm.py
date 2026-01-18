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
def tanhm(A):
    """
    Compute the hyperbolic matrix tangent.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array

    Returns
    -------
    tanhm : (N, N) ndarray
        Hyperbolic matrix tangent of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanhm, sinhm, coshm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> t = tanhm(a)
    >>> t
    array([[ 0.3428582 ,  0.51987926],
           [ 0.17329309,  0.86273746]])

    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

    >>> s = sinhm(a)
    >>> c = coshm(a)
    >>> t - s.dot(np.linalg.inv(c))
    array([[  2.72004641e-15,   4.55191440e-15],
           [  0.00000000e+00,  -5.55111512e-16]])

    """
    A = _asarray_square(A)
    return _maybe_real(A, solve(coshm(A), sinhm(A)))