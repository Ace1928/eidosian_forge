from __future__ import annotations
from pyomo.common.dependencies import mpi4py
from .mpi_block_vector import MPIBlockVector
from .block_vector import BlockVector
from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
from .block_matrix import assert_block_structure as block_matrix_assert_block_structure
from .base_block import BaseBlockMatrix
import numpy as np
from scipy.sparse import coo_matrix
import operator
@property
def owned_blocks(self):
    """
        Returns list with inidices of blocks owned by this processor.
        """
    return list(zip(*np.nonzero(self._owned_mask)))