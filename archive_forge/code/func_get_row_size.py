from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def get_row_size(self, row):
    if row in self._undefined_brows:
        raise NotFullyDefinedBlockMatrixError('The dimensions of the requested row are not defined.')
    return int(self._brow_lengths[row])