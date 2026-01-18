from ._base import issparse
from ._csr import csr_array
from ._sparsetools import csr_count_blocks
def count_blocks(A, blocksize):
    """For a given blocksize=(r,c) count the number of occupied
    blocks in a sparse matrix A
    """
    r, c = blocksize
    if r < 1 or c < 1:
        raise ValueError('r and c must be positive')
    if issparse(A):
        if A.format == 'csr':
            M, N = A.shape
            return csr_count_blocks(M, N, r, c, A.indptr, A.indices)
        elif A.format == 'csc':
            return count_blocks(A.T, (c, r))
    return count_blocks(csr_array(A), blocksize)