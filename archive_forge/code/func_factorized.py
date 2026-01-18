import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
def factorized(A):
    """Return a function for solving a sparse linear system, with A pre-factorized.

    Args:
        A (cupyx.scipy.sparse.spmatrix): Sparse matrix to factorize.

    Returns:
        callable: a function to solve the linear system of equations given in
        ``A``.

    Note:
        This function computes LU decomposition of a sparse matrix on the CPU
        using `scipy.sparse.linalg.splu`. Therefore, LU decomposition is not
        accelerated on the GPU. On the other hand, the computation of solving
        linear equations using the method returned by this function is
        performed on the GPU.

    .. seealso:: :func:`scipy.sparse.linalg.factorized`
    """
    return splu(A).solve