import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def aslinearoperator(A):
    """Return `A` as a LinearOperator.

    Args:
        A (array-like):
            The input array to be converted to a `LinearOperator` object.
            It may be any of the following types:

               * :class:`cupy.ndarray`
               * sparse matrix (e.g. ``csr_matrix``, ``coo_matrix``, etc.)
               * :class:`cupyx.scipy.sparse.linalg.LinearOperator`
               * object with ``.shape`` and ``.matvec`` attributes

    Returns:
        cupyx.scipy.sparse.linalg.LinearOperator: `LinearOperator` object

    .. seealso:: :func:`scipy.sparse.aslinearoperator``
    """
    if isinstance(A, LinearOperator):
        return A
    elif isinstance(A, cupy.ndarray):
        if A.ndim > 2:
            raise ValueError('array must have ndim <= 2')
        A = cupy.atleast_2d(A)
        return MatrixLinearOperator(A)
    elif sparse.isspmatrix(A):
        return MatrixLinearOperator(A)
    elif hasattr(A, 'shape') and hasattr(A, 'matvec'):
        rmatvec = None
        rmatmat = None
        dtype = None
        if hasattr(A, 'rmatvec'):
            rmatvec = A.rmatvec
        if hasattr(A, 'rmatmat'):
            rmatmat = A.rmatmat
        if hasattr(A, 'dtype'):
            dtype = A.dtype
        return LinearOperator(A.shape, A.matvec, rmatvec=rmatvec, rmatmat=rmatmat, dtype=dtype)
    else:
        raise TypeError('type not understood')