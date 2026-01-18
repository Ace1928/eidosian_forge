from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def copyfrom(self, other, deep=True):
    """
        Copies entries of other matrix into this matrix. This method provides
        an easy way to populate a BlockMatrix from scipy.sparse matrices. It also
        intended to facilitate copying values from other BlockMatrix to this BlockMatrix

        Parameters
        ----------
        other: BlockMatrix or scipy.spmatrix
        deep: bool
            If deep is True and other is a BlockMatrix, then the blocks in other are copied. If deep is False
            and other is a BlockMatrix, then the blocks in other are not copied.

        Returns
        -------
        None

        """
    assert_block_structure(self)
    if isinstance(other, BlockMatrix):
        assert other.bshape == self.bshape, 'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
        m, n = self.bshape
        if deep:
            for i in range(m):
                for j in range(n):
                    if not other.is_empty_block(i, j):
                        self.set_block(i, j, other.get_block(i, j).copy())
                    else:
                        self.set_block(i, j, None)
        else:
            for i in range(m):
                for j in range(n):
                    self.set_block(i, j, other.get_block(i, j))
    elif isspmatrix(other) or isinstance(other, np.ndarray):
        assert other.shape == self.shape, 'dimensions mismatch {} != {}'.format(self.shape, other.shape)
        if isinstance(other, np.ndarray):
            m = csr_matrix(other)
        else:
            m = other.tocsr()
        row_offsets = np.append(0, np.cumsum(self._brow_lengths))
        col_offsets = np.append(0, np.cumsum(self._bcol_lengths))
        for i in range(self.bshape[0]):
            mm = m[row_offsets[i]:row_offsets[i + 1], :].tocsc()
            for j in range(self.bshape[1]):
                mmm = mm[:, col_offsets[j]:col_offsets[j + 1]]
                if self.is_empty_block(i, j) and mmm.nnz == 0:
                    self.set_block(i, j, None)
                else:
                    self.set_block(i, j, mmm)
    else:
        raise NotImplementedError('Format not supported. BlockMatrix can only copy data from another BlockMatrix, a numpy array, or a scipy sparse matrix.')