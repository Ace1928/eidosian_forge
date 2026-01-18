from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def col_block_sizes(self, copy=True):
    """
        Returns array with col-block sizes

        Parameters
        ----------
        copy: bool
            If False, then the internal array which stores the column block sizes will be returned without being copied.
            Setting copy to False is risky and should only be done with extreme care.

        Returns
        -------
        numpy.ndarray

        """
    if self.has_undefined_col_sizes():
        raise NotFullyDefinedBlockMatrixError('Some block column lengths are not defined: {0}'.format(str(self._bcol_lengths)))
    if copy:
        return self._bcol_lengths.copy()
    else:
        return self._bcol_lengths