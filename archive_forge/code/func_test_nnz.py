import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_nnz(self):
    self.assertEqual(self.square_mpi_mat.nnz, 12)
    self.assertEqual(self.square_mpi_mat2.nnz, 18)
    self.assertEqual(self.rectangular_mpi_mat.nnz, 16)