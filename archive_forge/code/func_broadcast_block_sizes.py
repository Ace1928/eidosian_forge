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
def broadcast_block_sizes(self):
    """
        Send sizes of all blocks to all processors. After this method is called
        this MPIBlockMatrix knows it's dimensions of all rows and columns. This method
        must be called before running any operations with the MPIBlockMatrix.
        """
    rank = self._mpiw.Get_rank()
    num_processors = self._mpiw.Get_size()
    local_row_data = np.zeros(self.bshape[0], dtype=np.int64)
    local_col_data = np.zeros(self.bshape[1], dtype=np.int64)
    local_row_data.fill(-1)
    local_col_data.fill(-1)
    for row_ndx in range(self.bshape[0]):
        if self._block_matrix.is_row_size_defined(row_ndx):
            local_row_data[row_ndx] = self._block_matrix.get_row_size(row_ndx)
    for col_ndx in range(self.bshape[1]):
        if self._block_matrix.is_col_size_defined(col_ndx):
            local_col_data[col_ndx] = self._block_matrix.get_col_size(col_ndx)
    send_data = np.concatenate([local_row_data, local_col_data])
    receive_data = np.empty(num_processors * (self.bshape[0] + self.bshape[1]), dtype=np.int64)
    self._mpiw.Allgather(send_data, receive_data)
    proc_dims = np.split(receive_data, num_processors)
    m, n = self.bshape
    brow_lengths = np.zeros(m, dtype=np.int64)
    bcol_lengths = np.zeros(n, dtype=np.int64)
    for i in range(m):
        rows_length = set()
        for k in range(num_processors):
            row_sizes, col_sizes = np.split(proc_dims[k], [self.bshape[0]])
            rows_length.add(row_sizes[i])
        if len(rows_length) > 2:
            msg = 'Row {} has more than one dimension across processors'.format(i)
            raise RuntimeError(msg)
        elif len(rows_length) == 2:
            if -1 not in rows_length:
                msg = 'Row {} has more than one dimension across processors'.format(i)
                raise RuntimeError(msg)
            rows_length.remove(-1)
        elif -1 in rows_length:
            msg = 'The dimensions of block row {} were not defined in any process'.format(i)
            raise NotFullyDefinedBlockMatrixError(msg)
        brow_lengths[i] = rows_length.pop()
    for i in range(n):
        cols_length = set()
        for k in range(num_processors):
            rows_sizes, col_sizes = np.split(proc_dims[k], [self.bshape[0]])
            cols_length.add(col_sizes[i])
        if len(cols_length) > 2:
            msg = 'Column {} has more than one dimension across processors'.format(i)
            raise RuntimeError(msg)
        elif len(cols_length) == 2:
            if -1 not in cols_length:
                msg = 'Column {} has more than one dimension across processors'.format(i)
                raise RuntimeError(msg)
            cols_length.remove(-1)
        elif -1 in cols_length:
            msg = 'The dimensions of block column {} were not defined in any process'.format(i)
            raise NotFullyDefinedBlockMatrixError(msg)
        bcol_lengths[i] = cols_length.pop()
    for row_ndx, row_size in enumerate(brow_lengths):
        self.set_row_size(row_ndx, row_size)
    for col_ndx, col_size in enumerate(bcol_lengths):
        self.set_col_size(col_ndx, col_size)