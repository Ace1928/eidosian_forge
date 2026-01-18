import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_get_block_column_index(self):
    m = self.square_mpi_mat
    self.assertEqual(m.get_block_column_index(0), 0)
    self.assertEqual(m.get_block_column_index(1), 0)
    self.assertEqual(m.get_block_column_index(2), 0)
    self.assertEqual(m.get_block_column_index(3), 0)
    self.assertEqual(m.get_block_column_index(4), 1)
    self.assertEqual(m.get_block_column_index(5), 1)
    self.assertEqual(m.get_block_column_index(6), 1)
    self.assertEqual(m.get_block_column_index(7), 1)