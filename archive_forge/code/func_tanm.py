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
def tanm(A):
    """
    Compute the matrix tangent.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array.

    Returns
    -------
    tanm : (N, N) ndarray
        Matrix tangent of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanm, sinm, cosm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> t = tanm(a)
    >>> t
    array([[ -2.00876993,  -8.41880636],
           [ -2.80626879, -10.42757629]])

    Verify tanm(a) = sinm(a).dot(inv(cosm(a)))

    >>> s = sinm(a)
    >>> c = cosm(a)
    >>> s.dot(np.linalg.inv(c))
    array([[ -2.00876993,  -8.41880636],
           [ -2.80626879, -10.42757629]])

    """
    A = _asarray_square(A)
    return _maybe_real(A, solve(cosm(A), sinm(A)))