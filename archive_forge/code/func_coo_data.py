from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def coo_data(self):
    """
        Returns data array of matrix. The array corresponds to
        the data pointer in COOrdinate matrix format.

        Returns
        -------
        numpy.ndarray with values of all entries in the matrix

        """
    assert_block_structure(self)
    nonzeros = self.nnz
    data = np.empty(nonzeros, dtype=self.dtype)
    nnz = 0
    ii, jj = np.nonzero(self._block_mask)
    for i, j in zip(ii, jj):
        B = self._blocks[i, j].tocoo()
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        nnz += B.nnz
    return data