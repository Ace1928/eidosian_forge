from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def get_block_column_index(self, index):
    """
        Returns block-column idx from matrix column index.

        Parameters
        ----------
        index: int
            Column index

        Returns
        -------
        int

        """
    msgc = 'Operation not allowed with None columns. Specify at least one block in every column'
    assert not self.has_undefined_col_sizes(), msgc
    bm, bn = self.bshape
    cum = self._bcol_lengths.cumsum()
    assert index >= 0, 'index out of bounds'
    assert index < cum[bn - 1], 'index out of bounds'
    if bn <= 1:
        return 0
    ge = cum >= index
    block_index = np.argmax(ge)
    if cum[block_index] == index:
        return block_index + 1
    return block_index