import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def issparse(x):
    """Checks if a given matrix is a sparse matrix.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.spmatrix` that is
        a base class of all sparse matrix classes.

    """
    return isinstance(x, spmatrix)