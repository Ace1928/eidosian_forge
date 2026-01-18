import numpy as np
import scipy.sparse as sp
def get_row_nnz(mat, row):
    """Return the number of nonzeros in row.
    """
    return mat.indptr[row + 1] - mat.indptr[row]