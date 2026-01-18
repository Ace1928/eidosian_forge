import numpy
import cupy
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils
def bmat(blocks, format=None, dtype=None):
    """Builds a sparse matrix from sparse sub-blocks

    Args:
        blocks (array_like):
            Grid of sparse matrices with compatible shapes.
            An entry of None implies an all-zero matrix.
        format ({'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional):
            The sparse format of the result (e.g. "csr").  By default an
            appropriate sparse matrix format is returned.
            This choice is subject to change.
        dtype (dtype, optional):
            The data-type of the output matrix.  If not given, the dtype is
            determined from that of `blocks`.
    Returns:
        bmat (sparse matrix)

    .. seealso:: :func:`scipy.sparse.bmat`

    Examples:
        >>> from cupy import array
        >>> from cupyx.scipy.sparse import csr_matrix, bmat
        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))
        >>> B = csr_matrix(array([[5.], [6.]]))
        >>> C = csr_matrix(array([[7.]]))
        >>> bmat([[A, B], [None, C]]).toarray()
        array([[1., 2., 5.],
               [3., 4., 6.],
               [0., 0., 7.]])
        >>> bmat([[A, None], [None, C]]).toarray()
        array([[1., 2., 0.],
               [3., 4., 0.],
               [0., 0., 7.]])

    """
    M = len(blocks)
    N = len(blocks[0])
    blocks_flat = []
    for m in range(M):
        for n in range(N):
            if blocks[m][n] is not None:
                blocks_flat.append(blocks[m][n])
    if len(blocks_flat) == 0:
        return _coo.coo_matrix((0, 0), dtype=dtype)
    if N == 1 and format in (None, 'csr') and all((isinstance(b, _csr.csr_matrix) for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 0)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    elif M == 1 and format in (None, 'csc') and all((isinstance(b, _csc.csc_matrix) for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 1)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    block_mask = numpy.zeros((M, N), dtype=bool)
    brow_lengths = numpy.zeros(M + 1, dtype=numpy.int64)
    bcol_lengths = numpy.zeros(N + 1, dtype=numpy.int64)
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                A = _coo.coo_matrix(blocks[i][j])
                blocks[i][j] = A
                block_mask[i][j] = True
                if brow_lengths[i + 1] == 0:
                    brow_lengths[i + 1] = A.shape[0]
                elif brow_lengths[i + 1] != A.shape[0]:
                    msg = 'blocks[{i},:] has incompatible row dimensions. Got blocks[{i},{j}].shape[0] == {got}, expected {exp}.'.format(i=i, j=j, exp=brow_lengths[i + 1], got=A.shape[0])
                    raise ValueError(msg)
                if bcol_lengths[j + 1] == 0:
                    bcol_lengths[j + 1] = A.shape[1]
                elif bcol_lengths[j + 1] != A.shape[1]:
                    msg = 'blocks[:,{j}] has incompatible row dimensions. Got blocks[{i},{j}].shape[1] == {got}, expected {exp}.'.format(i=i, j=j, exp=bcol_lengths[j + 1], got=A.shape[1])
                    raise ValueError(msg)
    nnz = sum((block.nnz for block in blocks_flat))
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks_flat]
        dtype = _sputils.upcast(*all_dtypes) if all_dtypes else None
    row_offsets = numpy.cumsum(brow_lengths)
    col_offsets = numpy.cumsum(bcol_lengths)
    shape = (row_offsets[-1], col_offsets[-1])
    data = cupy.empty(nnz, dtype=dtype)
    idx_dtype = _sputils.get_index_dtype(maxval=max(shape))
    row = cupy.empty(nnz, dtype=idx_dtype)
    col = cupy.empty(nnz, dtype=idx_dtype)
    nnz = 0
    ii, jj = numpy.nonzero(block_mask)
    for i, j in zip(ii, jj):
        B = blocks[int(i)][int(j)]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = B.row + row_offsets[i]
        col[idx] = B.col + col_offsets[j]
        nnz += B.nnz
    return _coo.coo_matrix((data, (row, col)), shape=shape).asformat(format)