import cupy
from cupy import _core
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _data
from cupyx.scipy.sparse import _util
def isspmatrix_dia(x):
    """Checks if a given matrix is of DIA format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.dia_matrix`.

    """
    return isinstance(x, dia_matrix)