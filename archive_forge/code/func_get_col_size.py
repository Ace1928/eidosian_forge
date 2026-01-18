from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def get_col_size(self, col):
    if col in self._undefined_bcols:
        raise NotFullyDefinedBlockMatrixError('The dimensions of the requested column are not defined.')
    return int(self._bcol_lengths[col])