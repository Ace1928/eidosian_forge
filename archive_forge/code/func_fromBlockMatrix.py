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
@staticmethod
def fromBlockMatrix(block_matrix, rank_ownership, mpi_comm, assert_correct_owners=False):
    """
        Creates a parallel MPIBlockMatrix from blockmatrix

        Parameters
        ----------
        block_matrix: BlockMatrix
            The block matrix to use to create the MPIBlockMatrix
        rank_ownership: array_like
            2D-array with processor ownership of each block. A block can be own by a
            single processor or by all processors. Blocks own by all processors have
            ownership -1. Blocks own by a single processor have ownership rank. where
            rank=MPI.COMM_WORLD.Get_rank()
        mpi_comm: MPI communicator
            An MPI communicator. Tyically MPI.COMM_WORLD
        """
    block_matrix_assert_block_structure(block_matrix)
    bm, bn = block_matrix.bshape
    mat = MPIBlockMatrix(bm, bn, rank_ownership, mpi_comm, assert_correct_owners=assert_correct_owners)
    for i in range(bm):
        mat.set_row_size(i, block_matrix.get_row_size(i))
    for j in range(bn):
        mat.set_col_size(j, block_matrix.get_col_size(j))
    for i, j in mat.owned_blocks:
        mat.set_block(i, j, block_matrix.get_block(i, j))
    return mat