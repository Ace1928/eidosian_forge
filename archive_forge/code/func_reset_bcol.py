from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def reset_bcol(self, jdx):
    """
        Resets all blocks in selected block-column to None

        Parameters
        ----------
        jdx: int
            block-column index to be reset

        Returns
        -------
        None

        """
    assert 0 <= jdx < self.bshape[1], 'Index out of bounds'
    self._block_mask[:, jdx] = False
    self._blocks[:, jdx] = None