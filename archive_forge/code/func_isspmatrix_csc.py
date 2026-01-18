import cupy
from cupyx import cusparse
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy.sparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
def isspmatrix_csc(x):
    """Checks if a given matrix is of CSC format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csc_matrix`.

    """
    return isinstance(x, csc_matrix)