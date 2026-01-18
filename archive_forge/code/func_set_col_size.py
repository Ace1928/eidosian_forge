from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def set_col_size(self, col, size):
    if col in self._undefined_bcols:
        self._undefined_bcols.remove(col)
        self._bcol_lengths[col] = size
        if len(self._undefined_bcols) == 0:
            self._bcol_lengths = np.asarray(self._bcol_lengths, dtype=np.int64)
    elif self._bcol_lengths[col] != size:
        raise ValueError('Incompatible column dimensions for column {col}; got {got}; expected {exp}'.format(col=col, got=size, exp=self._bcol_lengths[col]))