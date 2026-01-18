from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def get_block_row_index(self, index):
    """
        Returns block-row idx from matrix row index.

        Parameters
        ----------
        index: int
            Row index

        Returns
        -------
        int

        """
    msgr = 'Operation not allowed with None rows. Specify at least one block in every row'
    assert not self.has_undefined_row_sizes(), msgr
    bm, bn = self.bshape
    cum = self._brow_lengths.cumsum()
    assert index >= 0, 'index out of bounds'
    assert index < cum[bm - 1], 'index out of bounds'
    if bm <= 1:
        return 0
    ge = cum >= index
    block_index = np.argmax(ge)
    if cum[block_index] == index:
        return block_index + 1
    return block_index