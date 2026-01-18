import numpy
import cupy
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils
def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """Construct a sparse matrix from diagonals.

    Args:
        diagonals (sequence of array_like):
            Sequence of arrays containing the matrix diagonals, corresponding
            to `offsets`.
        offsets (sequence of int or an int):
            Diagonals to set:
                - k = 0  the main diagonal (default)
                - k > 0  the k-th upper diagonal
                - k < 0  the k-th lower diagonal
        shape (tuple of int):
            Shape of the result. If omitted, a square matrix large enough
            to contain the diagonals is returned.
        format ({"dia", "csr", "csc", "lil", ...}):
            Matrix format of the result.  By default (format=None) an
            appropriate sparse matrix format is returned.  This choice is
            subject to change.
        dtype (dtype): Data type of the matrix.

    Returns:
        cupyx.scipy.sparse.spmatrix: Generated matrix.

    Notes:
        This function differs from `spdiags` in the way it handles
        off-diagonals.

        The result from `diags` is the sparse equivalent of::

            cupy.diag(diagonals[0], offsets[0])
            + ...
            + cupy.diag(diagonals[k], offsets[k])

        Repeated diagonal offsets are disallowed.
    """
    if _sputils.isscalarlike(offsets):
        if len(diagonals) == 0 or _sputils.isscalarlike(diagonals[0]):
            diagonals = [cupy.atleast_1d(diagonals)]
        else:
            raise ValueError('Different number of diagonals and offsets.')
    else:
        diagonals = list(map(cupy.atleast_1d, diagonals))
    if isinstance(offsets, cupy.ndarray):
        offsets = offsets.get()
    offsets = numpy.atleast_1d(offsets)
    if len(diagonals) != len(offsets):
        raise ValueError('Different number of diagonals and offsets.')
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)
    if dtype is None:
        dtype = cupy.common_type(*diagonals)
    m, n = shape
    M = max([min(m + offset, n - offset) + max(0, offset) for offset in offsets])
    M = max(0, M)
    data_arr = cupy.zeros((len(offsets), M), dtype=dtype)
    K = min(m, n)
    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError('Offset %d (index %d) out of bounds' % (offset, j))
        try:
            data_arr[j, k:k + length] = diagonal[..., :length]
        except ValueError:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError('Diagonal length (index %d: %d at offset %d) does not agree with matrix size (%d, %d).' % (j, len(diagonal), offset, m, n))
            raise
    return _dia.dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)