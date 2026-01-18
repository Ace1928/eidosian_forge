import numpy
import cupy
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils
def spdiags(data, diags, m, n, format=None):
    """Creates a sparse matrix from diagonals.

    Args:
        data (cupy.ndarray): Matrix diagonals stored row-wise.
        diags (cupy.ndarray): Diagonals to set.
        m (int): Number of rows.
        n (int): Number of cols.
        format (str or None): Sparse format, e.g. ``format="csr"``.

    Returns:
        cupyx.scipy.sparse.spmatrix: Created sparse matrix.

    .. seealso:: :func:`scipy.sparse.spdiags`

    """
    return _dia.dia_matrix((data, diags), shape=(m, n)).asformat(format)