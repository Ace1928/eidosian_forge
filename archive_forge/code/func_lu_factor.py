from warnings import warn
import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray
@_uarray.implements('lu_factor')
def lu_factor(a, overwrite_a=False, check_finite=True):
    """LU decomposition.

    Decompose a given two-dimensional square matrix into ``P * L * U``,
    where ``P`` is a permutation matrix,  ``L`` lower-triangular with
    unit diagonal elements, and ``U`` upper-triangular matrix.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(M, N)``
        overwrite_a (bool): Allow overwriting data in ``a`` (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        tuple:
            ``(lu, piv)`` where ``lu`` is a :class:`cupy.ndarray`
            storing ``U`` in its upper triangle, and ``L`` without
            unit diagonal elements in its lower triangle, and ``piv`` is
            a :class:`cupy.ndarray` storing pivot indices representing
            permutation matrix ``P``. For ``0 <= i < min(M,N)``, row
            ``i`` of the matrix was interchanged with row ``piv[i]``

    .. seealso:: :func:`scipy.linalg.lu_factor`
    """
    return _lu_factor(a, overwrite_a, check_finite)