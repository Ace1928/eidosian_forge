import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_is_row_size_defined(self):
    m = MPIBlockMatrix(2, 2, [[0, 1], [-1, -1]], comm)
    rank = comm.Get_rank()
    m0 = np.array([[4.3, 0], [-2.1, 1.5]])
    m0 = coo_matrix(m0)
    m1 = np.array([[0, 0], [0, 3.2]])
    m1 = coo_matrix(m1)
    if rank == 0:
        m.set_block(0, 0, m0)
    elif rank == 1:
        m.set_block(0, 1, m1)
    self.assertTrue(m.is_row_size_defined(0, False))
    self.assertFalse(m.is_row_size_defined(1, False))
    self.assertTrue(m.is_col_size_defined(0, False))
    self.assertTrue(m.is_col_size_defined(1, False))
    m = MPIBlockMatrix(2, 2, [[0, -1], [1, -1]], comm)
    rank = comm.Get_rank()
    m0 = np.array([[4.3, 0], [-2.1, 1.5]])
    m0 = coo_matrix(m0)
    m1 = np.array([[0, 0], [0, 3.2]])
    m1 = coo_matrix(m1)
    if rank == 0:
        m.set_block(0, 0, m0)
    elif rank == 1:
        m.set_block(1, 0, m1)
    self.assertTrue(m.is_row_size_defined(0, False))
    self.assertTrue(m.is_row_size_defined(1, False))
    self.assertTrue(m.is_col_size_defined(0, False))
    self.assertFalse(m.is_col_size_defined(1, False))