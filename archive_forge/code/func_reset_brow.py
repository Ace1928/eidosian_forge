from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def reset_brow(self, idx):
    """
        Resets all blocks in selected block-row to None

        Parameters
        ----------
        idx: int
            block-row index to be reset

        Returns
        -------
        None

        """
    assert 0 <= idx < self.bshape[0], 'Index out of bounds'
    self._block_mask[idx, :] = False
    self._blocks[idx, :] = None